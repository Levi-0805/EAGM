"""New tool--fixture contact task; historical XML, controllers and records untouched.

The 14-joint robot has a rigid peg, not a grasping policy. Robot self/environment
contacts are disabled; the peg and fixture alone participate in contact. This
restricted task must not be presented as general bimanual assembly.
"""
from __future__ import annotations
import hashlib
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from controllers.partial_unknown import PartialUnknownArmEnv, ControllerBConfig
from controllers.compatible_arm_env import ArmEnv, PILOT_EFFECTIVE, LOCAL_MESHDIR

ROOT=Path(__file__).resolve().parents[1]
HOME=np.array([0,-.25,0,.5,0,0,0,0,.25,0,.5,0,0,0],float)

def _numbers(x): return ' '.join(f'{v:.16g}' for v in np.asarray(x).ravel())

def build_contact_asset():
    root=ET.fromstring(PILOT_EFFECTIVE.read_text())
    root.find('compiler').set('meshdir',str(LOCAL_MESHDIR))
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    head=visual.find('headlight')
    if head is None:head=ET.SubElement(visual,'headlight')
    head.set('ambient','.4 .4 .4');head.set('diffuse','.8 .8 .8');head.set('active','1')
    for g in root.iter('geom'):
        g.set('contype','0');g.set('conaffinity','0')
    body=root.find(".//body[@name='openarm_left_ee_base_link']")
    ET.SubElement(body,'geom',name='eagm2_peg',type='cylinder',fromto='.05 0 0 .11 0 0',size='.004',mass='.02',rgba='.9 .35 .1 1',contype='2',conaffinity='2',friction='.5 .01 .001',solref='.006 1')
    ET.SubElement(body,'site',name='eagm2_peg_tip',pos='.11 0 0',size='.002',rgba='1 0 0 1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    data.qpos[:]=HOME;mujoco.mj_forward(model,data)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'openarm_left_ee_base_link')
    rot=data.xmat[bid].reshape(3,3).copy(); tip=data.xpos[bid]+rot@np.array([.11,0,0])
    # Fixture mouth starts 15 mm in front of the nominal home tip.
    mouth=tip+rot[:,0]*.015
    frame=np.column_stack([rot[:,1],rot[:,2],rot[:,0]])
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,frame.ravel())
    fixture=ET.SubElement(root.find('worldbody'),'body',name='eagm2_fixture',pos=_numbers(mouth),quat=_numbers(quat))
    # Square aperture 10 mm wide around an 8 mm diameter round peg.
    for name,pos,size in [('xp',[.013,0,.025],[.008,.021,.025]),('xm',[-.013,0,.025],[.008,.021,.025]),('yp',[0,.013,.025],[.005,.008,.025]),('ym',[0,-.013,.025],[.005,.008,.025]),('bottom',[0,0,.038],[.021,.021,.003])]:
        ET.SubElement(fixture,'geom',name='eagm2_fixture_'+name,type='box',pos=_numbers(pos),size=_numbers(size),rgba='.2 .45 .65 1',contype='2',conaffinity='2',friction='.5 .01 .001',solref='.006 1')
    ET.SubElement(fixture,'site',name='eagm2_hole_mouth',pos='0 0 0',size='.002',rgba='0 1 0 1')
    text=ET.tostring(root,encoding='unicode');sha=hashlib.sha256(text.encode()).hexdigest()
    path=ROOT/'eagm2/assets'/f'contact_{sha[:16]}.xml'
    if not path.exists():path.write_text(text)
    return path,mouth,frame,rot

class ContactEnv(PartialUnknownArmEnv):
    """Controller B with a new strict-model-hash snapshot and task metrics."""
    def __init__(self,config=None,delay=0,alpha=.5):
        path,self.mouth,self.fixture_rotation,self.goal_rotation=build_contact_asset()
        self.b_config=config or ControllerBConfig(gain=1.05,extra_lag=1,rate_limit=.045,smooth_beta=.25)
        self._active={k:getattr(self.b_config,k) for k in ('gain','extra_lag','rate_limit','smooth_beta')}
        self._extra_queue=[];self._renderer=None;self._ready=False
        self._held_seconds=0.;self._success=False;self._completion_time=None;self._contacts=[];self._forces=[];self._penetrations=[]
        ArmEnv.__init__(self,delay=delay,alpha=alpha,model_path=path,enable_contact=True)
        self._tip_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'eagm2_peg_tip')
        self._body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'openarm_left_ee_base_link')
        self._peg_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'eagm2_peg')
        self._fk_data=mujoco.MjData(self.model)
        self._ready=True
        self.q_insert=self._ik(HOME,self.mouth+self.fixture_rotation[:,2]*.036)
        self.q_align=self._ik(HOME,self.mouth-self.fixture_rotation[:,2]*.005)
        self.reset(seed=0,delay=delay,jitter=0)

    def reset(self,seed=None,delay=None,pose=None,jitter=.002):
        x=PartialUnknownArmEnv.reset(self,seed,delay,pose,jitter)
        self._held_seconds=0.;self._success=False;self._completion_time=None;self._contacts=[];self._forces=[];self._penetrations=[]
        return x

    def _pose(self,q=None):
        d=self._fk_data
        d.qpos[:]=self.data.qpos if q is None else q
        mujoco.mj_fwdPosition(self.model,d)
        return d.site_xpos[self._tip_id].copy(),d.xmat[self._body_id].reshape(3,3).copy()

    def _ik(self,q0,target):
        q=q0.copy();d=self._fk_data
        for _ in range(160):
            p,R=self._pose(q)
            er=.5*sum(np.cross(R[:,i],self.goal_rotation[:,i]) for i in range(3))
            err=np.r_[target-p,.15*er]
            if np.linalg.norm(err)<1e-7:break
            jp=np.zeros((3,self.model.nv));jr=jp.copy()
            mujoco.mj_jacSite(self.model,d,jp,jr,self._tip_id)
            J=np.vstack([jp[:,:7],.15*jr[:,:7]])
            dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),err)
            q[:7]=np.clip(q[:7]+np.clip(dq,-.05,.05),self.lower[:7],self.upper[:7])
        return q

    def reference(self,horizon,start_frame=0):
        refs=[]
        for f in range(start_frame,start_frame+horizon):
            t=f*self.control_dt
            if t<1.:a=np.clip(t,0,1);q=(1-a)*HOME+a*self.q_align
            else:a=np.clip((t-1.)/3.,0,1);q=(1-a)*self.q_align+a*self.q_insert
            refs.append(q)
        return np.asarray(refs)

    def geometry(self,q=None):
        p,R=self._pose(q);local=self.fixture_rotation.T@(p-self.mouth)
        angle=np.arccos(np.clip(R[:,0]@self.fixture_rotation[:,2],-1,1))
        return dict(depth=float(local[2]),lateral_error=float(np.linalg.norm(local[:2])),angle_rad=float(angle),tip=p)

    def step(self,action):
        x=PartialUnknownArmEnv.step(self,action)
        g=self.geometry();n=0;force=0.;penetration=0.
        for i in range(self.data.ncon):
            con=self.data.contact[i]
            if self._peg_id in (con.geom1,con.geom2):
                n+=1;w=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,w)
                force+=float(np.linalg.norm(w[:3]));penetration=max(penetration,max(0.,-float(con.dist)))
        self._contacts.append(n>0);self._forces.append(force);self._penetrations.append(penetration)
        # Success cannot be earned by penetrating through the bottom/side wall.
        within=g['depth']>=.020 and g['depth']<=.0365 and g['lateral_error']<=.002 and g['angle_rad']<=np.deg2rad(5) and penetration<=.001
        self._held_seconds=self._held_seconds+self.control_dt if within else 0.
        if not self._success and self._held_seconds>=.5 and self.frame*self.control_dt<=10.:
            self._success=True;self._completion_time=self.frame*self.control_dt
        return x

    def reward(self,action,prev):
        g=self.geometry();qv=self.observe()[14:]
        # Continuous physical geometry reward; terminal success separate.
        return float(-100*g['lateral_error']**2-20*(g['depth']-.030)**2-.04*(1-np.cos(g['angle_rad']))-.0005*np.mean(qv*qv)-.05*np.mean((np.asarray(action)-prev)**2))

    def metrics(self):
        g=self.geometry();return dict(success=bool(self._success),completion_time=float(self._completion_time or 10.),elapsed=self.frame*self.control_dt,depth=g['depth'],position_error=g['lateral_error'],angle_deg=np.rad2deg(g['angle_rad']),contact_fraction=float(np.mean(self._contacts)) if self._contacts else 0.,contact_force_std=float(np.std(self._forces)) if self._forces else 0.,max_penetration=max(self._penetrations,default=0.),held_seconds=self._held_seconds)

    def snapshot(self):
        snap=PartialUnknownArmEnv.snapshot(self)
        snap['contact_task']={k:deepcopy(getattr(self,k)) for k in ['_held_seconds','_success','_completion_time','_contacts','_forces','_penetrations']}
        return snap

    def restore(self,snapshot):
        if np.asarray(snapshot['integration']).shape!=(self._state_size,):raise ValueError('Snapshot state size mismatch')
        if 'controller_b' not in snapshot or 'contact_task' not in snapshot:raise ValueError('Complete contact snapshot required')
        x=ArmEnv.restore(self,snapshot)
        b=snapshot['controller_b'];self._active=dict(b['active']);self._extra_queue=[np.array(v,copy=True) for v in b['extra_queue']]
        for key in ['u_smooth','u_prev','last_true_command','last_nominal_command']:
            setattr(self,'_'+key if key.startswith('u_') else key,np.array(b[key],copy=True))
        for k,v in snapshot['contact_task'].items():setattr(self,k,deepcopy(v))
        return x

    def render_rgb(self,size=64,camera_shift=None):
        if self._renderer is None or self._renderer.width!=size:
            if self._renderer is not None:self._renderer.close()
            self._renderer=mujoco.Renderer(self.model,height=size,width=size)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=self.mouth+(np.zeros(3) if camera_shift is None else np.asarray(camera_shift))
        eye=self.fixture_rotation@np.array([.12,.08,-.18])
        cam.distance=float(np.linalg.norm(eye));cam.azimuth=float(np.rad2deg(np.arctan2(eye[1],eye[0])));cam.elevation=float(np.rad2deg(np.arcsin(eye[2]/cam.distance)))
        # Refresh derived geometry in scratch data, never disturb live integrator/warm-start.
        if not hasattr(self,'_render_data'):self._render_data=mujoco.MjData(self.model)
        integ=np.empty(self._state_size);mujoco.mj_getState(self.model,self.data,integ,self._spec)
        mujoco.mj_setState(self.model,self._render_data,integ,self._spec);mujoco.mj_forward(self.model,self._render_data)
        self._renderer.update_scene(self._render_data,camera=cam)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:self._renderer.close();self._renderer=None

class TorchContactReward(__import__('torch').nn.Module):
    """Exact kinematic transforms from the current MuJoCo model, not fitted FK.

    State-task only. Must never be passed true q as a visual-policy deploy input.
    """
    def __init__(self,env):
        import torch
        super().__init__();self.chain=[]
        ids=[];b=env._body_id
        while b>0:ids.append(b);b=int(env.model.body_parentid[b])
        for i,b in enumerate(reversed(ids)):
            R=np.empty(9);mujoco.mju_quat2Mat(R,env.model.body_quat[b])
            self.register_buffer(f'p{i}',torch.tensor(env.model.body_pos[b],dtype=torch.float64))
            self.register_buffer(f'R{i}',torch.tensor(R.reshape(3,3),dtype=torch.float64))
            joints=[]
            for j in range(env.model.body_jntadr[b],env.model.body_jntadr[b]+env.model.body_jntnum[b]):
                if env.model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('Expected hinge-only OpenArm chain')
                self.register_buffer(f'axis{j}',torch.tensor(env.model.jnt_axis[j],dtype=torch.float64))
                self.register_buffer(f'jp{j}',torch.tensor(env.model.jnt_pos[j],dtype=torch.float64))
                joints.append((j,int(env.model.jnt_qposadr[j]),float(env.model.qpos0[env.model.jnt_qposadr[j]])))
            self.chain.append((i,joints))
        self.register_buffer('mouth',torch.tensor(env.mouth,dtype=torch.float64))
        self.register_buffer('frame',torch.tensor(env.fixture_rotation,dtype=torch.float64))
        self.register_buffer('tip_offset',torch.tensor([.11,0,0],dtype=torch.float64))
    def geometry(self,q):
        import torch
        batch=q.shape[:-1];eye=torch.eye(3,dtype=q.dtype,device=q.device)
        R=eye.expand(*batch,3,3);p=torch.zeros(*batch,3,dtype=q.dtype,device=q.device)
        for i,joints in self.chain:
            bp=getattr(self,f'p{i}').to(q);br=getattr(self,f'R{i}').to(q)
            p=p+(R@bp.unsqueeze(-1)).squeeze(-1)
            R=R@br
            for j,qidx,qref in joints:
                axis=getattr(self,f'axis{j}').to(q);jp=getattr(self,f'jp{j}').to(q)
                x,y,z=axis.unbind();zero=x*0
                K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
                th=q[...,qidx]-qref;co=th.cos()[...,None,None];si=th.sin()[...,None,None]
                J=co*eye+(1-co)*axis[:,None]*axis[None,:]+si*K
                p=p+(R@(jp-(J@jp.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
                R=R@J
        tip=p+(R@self.tip_offset.to(q).unsqueeze(-1)).squeeze(-1)
        local=(tip-self.mouth.to(q))@self.frame.to(q)
        dot=(R[...,0]*self.frame.to(q)[:,2]).sum(-1)
        return local,dot
    def forward(self,x,a,prev):
        local,dot=self.geometry(x[...,:14])
        return -100*local[...,:2].square().sum(-1)-20*(local[...,2]-.030).square()-.04*(1-dot)-.0005*x[...,14:].square().mean(-1)-.05*(a-prev).square().mean(-1)
