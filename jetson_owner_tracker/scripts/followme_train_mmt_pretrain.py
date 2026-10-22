#!/usr/bin/env python3
"""MMT pre-training of IBN-ResNet-50 on MSMT17 (the paper's Re-ID network, Sect. VI).

Re-implements the *source-domain pre-training* stage of Mutual Mean-Teaching
(Ge et al., ICLR 2020, github.com/yxgeee/MMT ``examples/source_pretrain.py``)
which is the only MMT stage FollowMe uses:

* network: ``ResNetIBN('50a')`` (ImageNet IBN-a init), last stride 1,
  ``feat`` Linear 2048->D + BN (D = 256 in the paper), classifier over IDs
* loss: label-smoothed cross-entropy (eps 0.1) on the classifier output
  + soft-margin batch-hard triplet (MMT ``SoftTripletLoss``, margin 0) on the
  2048-d pooled feature
* sampler: P identities x K=4 images, batch 64
* augmentation: resize 256x128, flip, pad 10 + random crop, random erasing 0.5
* Adam lr 3.5e-4, weight decay 5e-4, 10-epoch linear warm-up, x0.1 at epochs
  40 and 70, 80 epochs x 200 iterations

The checkpoint (``{'state_dict': ...}``) loads with ``reid.backend: mmt``.

MSMT17 layout (MSMT17_V1 or V2)::

    root/train/<pid>/<img>.jpg      root/list_train.txt, root/list_val.txt  ("train/0000/xx.jpg 0")

    python scripts/followme_train_mmt_pretrain.py --msmt17 /data/MSMT17_V1 --out models/mmt_msmt17_ibn50a_256.pth.tar
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IBN_A_URL = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"


def load_msmt17(root: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for name in ("list_train.txt", "list_val.txt"):       # MMT trains on train + val
        f = root / name
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                rel, pid = line.split()
                items.append((str(root / rel), int(pid)))
    if not items:                                         # fallback: root/train/<pid>/*.jpg
        for d in sorted((root / "train").iterdir()):
            items += [(str(p), int(d.name)) for p in d.glob("*.jpg")]
    pids = sorted({p for _, p in items})
    relabel = {p: i for i, p in enumerate(pids)}
    return [(f, relabel[p]) for f, p in items]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Sampler
    from torchvision import transforms as T

    from followme.reid import build_mmt_ibn_resnet50

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msmt17", required=True)
    ap.add_argument("--out", default="models/mmt_msmt17_ibn50a_256.pth.tar")
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-instances", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[40, 70])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imagenet-ibn", default=IBN_A_URL, help="IBN-a ImageNet weights (path or URL)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_msmt17(Path(args.msmt17))
    num_classes = max(p for _, p in items) + 1
    print(f"MSMT17: {len(items)} images, {num_classes} identities")

    tf = T.Compose([T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                    T.Pad(10), T.RandomCrop((256, 128)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    T.RandomErasing(p=0.5, value=[0.485, 0.456, 0.406])])

    class DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            f, p = items[i]
            return tf(Image.open(f).convert("RGB")), p

    class PKSampler(Sampler):
        """P identities x K instances per batch (MMT RandomMultipleGallerySampler, simplified)."""

        def __init__(self, n_batches: int):
            self.index = defaultdict(list)
            for i, (_, p) in enumerate(items):
                self.index[p].append(i)
            self.pids = list(self.index)
            self.n = n_batches

        def __len__(self):
            return self.n * args.batch_size

        def __iter__(self):
            p_per = args.batch_size // args.num_instances
            for _ in range(self.n):
                for pid in random.sample(self.pids, p_per):
                    pool = self.index[pid]
                    pick = (random.sample(pool, args.num_instances) if len(pool) >= args.num_instances
                            else random.choices(pool, k=args.num_instances))
                    yield from pick

    loader = DataLoader(DS(), batch_size=args.batch_size, sampler=PKSampler(args.iters * args.epochs),
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = build_mmt_ibn_resnet50(args.features)
    # ImageNet IBN-a initialisation (conv1 -> base.0, bn1 -> base.1, layerN -> base.(3+N))
    src = (torch.hub.load_state_dict_from_url(args.imagenet_ibn, map_location="cpu")
           if args.imagenet_ibn.startswith("http") else torch.load(args.imagenet_ibn, map_location="cpu"))
    remap = {"conv1": "base.0", "bn1": "base.1", "layer1": "base.4", "layer2": "base.5",
             "layer3": "base.6", "layer4": "base.7"}
    init = {}
    for k, v in src.items():
        head, _, rest = k.partition(".")
        if head in remap:
            init[f"{remap[head]}.{rest}"] = v
    missing, _ = model.load_state_dict(init, strict=False)
    print(f"ImageNet init: {len(init)} tensors, new layers: {[m for m in missing if not m.startswith('base')]}")
    model.feat_bn.bias.requires_grad_(False)
    classifier = nn.Linear(model.num_features, num_classes, bias=False)
    nn.init.normal_(classifier.weight, std=0.001)
    if model.has_embedding:
        nn.init.kaiming_normal_(model.feat.weight, mode="fan_out")
        nn.init.constant_(model.feat.bias, 0)
    nn.init.constant_(model.feat_bn.weight, 1)
    nn.init.constant_(model.feat_bn.bias, 0)
    model.to(dev).train()
    classifier.to(dev)

    def forward_train(x):
        pooled = model.gap(model.base(x)).flatten(1)
        bn_x = model.feat_bn(model.feat(pooled)) if model.has_embedding else model.feat_bn(pooled)
        if model.has_embedding:
            bn_x = F.relu(bn_x)
        return pooled, classifier(bn_x)

    def ce_label_smooth(logits, targets, eps=0.1):
        logp = F.log_softmax(logits, 1)
        onehot = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1)
        onehot = (1 - eps) * onehot + eps / logits.size(1)
        return (-onehot * logp).mean(0).sum()

    def soft_triplet(emb, targets):
        d = torch.cdist(emb, emb).clamp(min=1e-12)
        same = targets.unsqueeze(0).eq(targets.unsqueeze(1)).float()
        ap = (d - 9999999.0 * (1 - same)).max(1).values
        an = (d + 9999999.0 * same).min(1).values
        return (-F.log_softmax(torch.stack([ap, an], 1), 1)[:, 1]).mean()   # margin = 0

    params = [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(epoch):
        warm = 0.01 + (1 - 0.01) * epoch / args.warmup if epoch < args.warmup else 1.0
        return warm * (0.1 ** sum(epoch >= m for m in args.milestones))

    it = iter(loader)
    for epoch in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_factor(epoch)
        t0, stats = time.time(), defaultdict(float)
        for _ in range(args.iters):
            x, y = next(it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            pooled, logits = forward_train(x)
            l_ce, l_tr = ce_label_smooth(logits, y), soft_triplet(pooled, y)
            loss = l_ce + l_tr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            stats["ce"] += l_ce.item()
            stats["tri"] += l_tr.item()
            stats["acc"] += (logits.argmax(1) == y).float().mean().item()
        print(f"epoch {epoch + 1}/{args.epochs} lr={opt.param_groups[0]['lr']:.2e} "
              + " ".join(f"{k}={v / args.iters:.3f}" for k, v in stats.items()) + f" ({time.time() - t0:.0f}s)")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, args.out)
    print(f"saved {args.out} (D={model.num_features})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
