"""Small conditional least-squares GAN in NumPy (Mao et al., 2017).

CPU implementation with explicit derivatives, checked against finite differences.
No neural-network runtime or GPU is required. This is a tabular feature GAN.
"""
import numpy as np


class Network:
    def __init__(self, dims, rng):
        self.params=[]
        for a,b in zip(dims[:-1],dims[1:]):
            self.params.extend([rng.normal(0,np.sqrt(2/a),(a,b)),np.zeros(b)])
        self.first=[np.zeros_like(p) for p in self.params]
        self.second=[np.zeros_like(p) for p in self.params]
        self.step=0

    def forward(self,x):
        cache=[]
        for layer in range(len(self.params)//2):
            w,b=self.params[2*layer:2*layer+2]
            pre=x@w+b;cache.append((x,pre))
            x=np.where(pre>=0,pre,.2*pre) if layer<len(self.params)//2-1 else pre
        return x,cache

    def backward(self,gradient,cache):
        grads=[None]*len(self.params)
        for layer in reversed(range(len(cache))):
            x,pre=cache[layer]
            if layer<len(cache)-1:gradient=gradient*np.where(pre>=0,1.,.2)
            grads[2*layer]=x.T@gradient;grads[2*layer+1]=gradient.sum(0)
            gradient=gradient@self.params[2*layer].T
        return gradient,grads

    def update(self,grads,lr=1e-4):
        self.step+=1
        norm=np.sqrt(sum(np.sum(g*g) for g in grads))
        scale=min(1.,10/max(norm,1e-12))
        for i,(p,g) in enumerate(zip(self.params,grads)):
            g=g*scale
            self.first[i]=.5*self.first[i]+.5*g
            self.second[i]=.9*self.second[i]+.1*g*g
            m=self.first[i]/(1-.5**self.step);v=self.second[i]/(1-.9**self.step)
            p-=lr*m/(np.sqrt(v)+1e-8)


def moment_loss_gradient(fake,real,weights):
    weights=np.asarray(weights)/max(np.mean(weights),1e-8)
    mu=fake.mean(0);variance=fake.var(0)
    dm=mu-real.mean(0);dv=variance-real.var(0)
    loss=np.mean(weights*(dm*dm+dv*dv))
    grad=weights[None,:]*(2*dm+4*dv*(fake-mu))/(fake.shape[0]*fake.shape[1])
    return float(loss),grad


def generate(z,indices,counts,rng,steps,batch_size,discriminator_steps,moment_weight,weights,gated):
    if steps<1 or batch_size<2 or discriminator_steps<1 or moment_weight<0:
        raise ValueError('Invalid GAN training settings')
    p,nc,latent=z.shape[1],len(indices),16
    generator=Network([latent+nc,64,64,p],rng)
    discriminator=Network([p+nc,64,64,1],rng)
    trace=[]
    def batch():
        cell=int(rng.integers(nc))
        cond=np.zeros((batch_size,nc));cond[:,cell]=1
        real=z[rng.choice(indices[cell],batch_size,replace=True)]
        return cell,cond,real
    for step in range(steps):
        for _ in range(discriminator_steps):
            cell,cond,real=batch()
            fake,_=generator.forward(np.c_[rng.normal(size=(batch_size,latent)),cond])
            dr,cr=discriminator.forward(np.c_[real,cond]);df,cf=discriminator.forward(np.c_[fake,cond])
            dloss=.5*np.mean((dr-1)**2)+.5*np.mean(df**2)
            _,gr=discriminator.backward((dr-1)/batch_size,cr)
            _,gf=discriminator.backward(df/batch_size,cf)
            discriminator.update([a+b for a,b in zip(gr,gf)])
        cell,cond,real=batch()
        fake,cg=generator.forward(np.c_[rng.normal(size=(batch_size,latent)),cond])
        df,cf=discriminator.forward(np.c_[fake,cond])
        adversarial=.5*np.mean((df-1)**2)
        through,_=discriminator.backward((df-1)/batch_size,cf)
        moment,mg=moment_loss_gradient(fake,z[indices[cell]],weights)
        gloss=adversarial+moment_weight*moment
        if not np.isfinite(gloss+dloss):raise ValueError('Nonfinite GAN loss')
        _,gg=generator.backward(through[:,:p]+moment_weight*mg,cg);generator.update(gg)
        if step%max(1,steps//20)==0 or step==steps-1:
            trace.append({'step':step+1,'discriminator':float(dloss),'generator':float(gloss),'moment':moment})
    candidates=[]
    for cell,n in enumerate(counts):
        n=n*5 if gated else n
        cond=np.zeros((n,nc));cond[:,cell]=1
        fake,_=generator.forward(np.c_[rng.normal(size=(n,latent)),cond]);candidates.append(fake)
    return candidates,trace
