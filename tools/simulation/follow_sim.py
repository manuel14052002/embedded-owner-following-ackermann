import math, numpy as np
P=dict(L=0.28,dmax=math.radians(30),follow=1.2,k_lin=0.8,vmax=0.24,vmin=0.168,vturn=0.199,turn_b=math.radians(75),
       dead=math.radians(3),k_ang=1.0,look=0.9,tol=0.10,acc=0.5,dec=1.0,rate=math.radians(90),
       trim_ki=0.4,trim_band=math.radians(12),trim_max=math.radians(6),trim_leak=30.0)
def clamp(x,a,b): return a if x<a else b if x>b else x
def owner_path(t,u,kind):
    if kind=="straight": return np.array([2.0+u*t,0.0])
    s=u*t  # L-turn after 4 m
    return np.array([2.0+s,0.0]) if s<4 else np.array([6.0,s-4])
def simulate(u=0.2,kind="straight",T=40.0,Tc=0.1,sigma=0.0,seed=0):
    rng=np.random.default_rng(seed); dt=0.01
    x=y=th=0.0; v=delta=trim=0.0; tnext=0.0; vcmd_d=0.0
    out=[]
    for i in range(int(T/dt)):
        t=i*dt; o=owner_path(t,u,kind)
        if t>=tnext-1e-9:
            # owner in robot frame (+ measurement noise)
            dx,dy=o[0]-x,o[1]-y; rx=math.cos(th)*dx+math.sin(th)*dy; ry=-math.sin(th)*dx+math.cos(th)*dy
            rx+=rng.normal(0,sigma); ry+=rng.normal(0,sigma)
            d=math.hypot(rx,ry); b=math.atan2(ry,rx); dg=d-P['follow']; h=Tc
            if dg<P['tol']: vd,dd=0.0,0.0
            elif abs(b)>P['turn_b']: vd,dd=P['vturn'],math.copysign(P['dmax'],b)
            else:
                vd=min(P['vmax'],P['k_lin']*dg)*max(0.0,math.cos(b))
                vd=0.0 if vd<1e-3 else max(vd,P['vmin'])
                dd=0.0 if (vd==0 or abs(b)<P['dead']) else clamp(math.atan(P['L']*P['k_ang']*2*math.sin(b)/max(dg,P['look'])),-P['dmax'],P['dmax'])
            trim*=max(0.0,1-h/P['trim_leak'])
            if v>0 and abs(b)<P['trim_band'] and dg>=P['tol']: trim=clamp(trim+P['trim_ki']*b*h,-P['trim_max'],P['trim_max'])
            if vd>0: dd=clamp(dd+trim,-P['dmax'],P['dmax'])
            v+=clamp(vd-v,-P['dec']*h,P['acc']*h)
            if vd>0 and 0<v<P['vmin']: v=P['vmin']
            step=P['rate']*h; delta=clamp(dd,delta-step,delta+step)
            tnext+=Tc
        x+=v*math.cos(th)*dt; y+=v*math.sin(th)*dt; th+=v*math.tan(delta)/P['L']*dt
        out.append((t,x,y,th,v,delta,o[0],o[1],math.hypot(o[0]-x,o[1]-y)))
    return np.array(out)
if __name__=="__main__":
    print("min turning radius L/tan(dmax) = %.3f m"%(P['L']/math.tan(P['dmax'])))
    for u in (0.15,0.20,0.24,0.30):
        a=simulate(u); dist=a[:,8]; late=dist[a[:,0]>25]
        print(f"straight u={u:.2f}: range at t>25 s mean {late.mean():.3f} min {late.min():.3f} max {late.max():.3f}; final {dist[-1]:.3f}; stops(v=0 share) {np.mean(a[a[:,0]>25,4]==0):.2f}")
    for Tc in (0.1,0.27):
        a=simulate(0.2,"L",T=45,Tc=Tc,sigma=0.03); dist=a[:,8]
        print(f"L-turn u=0.20 Tc={Tc}: range max {dist.max():.3f} final {dist[-1]:.3f}  max steer {math.degrees(np.abs(a[:,5]).max()):.1f} deg")
