// Original procedural geometry and analytic GGX teaching renderer. No model inference.
const canvas = document.querySelector('#material-canvas');
const status = document.querySelector('#gpu-status');
const controls = ['roughness', 'metalness', 'light-angle', 'material-hue'];
const state = { yaw: .45, pitch: .28, distance: 4.4, scene: 0, dragging: false };
const shader = `
struct Params { size: vec4f, camera: vec4f, material: vec4f, extra: vec4f };
@group(0) @binding(0) var<uniform> u: Params;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3)); return vec4f(p[i],0,1);
}
fn capsule(p:vec3f,a:vec3f,b:vec3f,r:f32)->f32 {
 let q=p-a;let v=b-a;return length(q-v*clamp(dot(q,v)/dot(v,v),0.,1.))-r;
}
fn object(p:vec3f)->f32 {
 if(u.extra.x>0.5){return length(p-vec3f(0,.08,0))-.83;}
 // Stylized original vessel, not the Utah dataset or experimental mesh.
 let body=(length(p/vec3f(.85,.64,.75))-1.)*.64;
 let lid=length((p-vec3f(0,.56,0))/vec3f(.55,.13,.49))*.13-.13;
 let knob=length(p-vec3f(0,.76,0))-.11;
 let spout=capsule(p,vec3f(.62,.05,0),vec3f(1.05,.53,0),.13);
 let h=p-vec3f(-.78,.12,0);let handle=length(vec2f((length(h.xy/vec2f(.48,.43))-1.)*.43,h.z))-.075;
 return min(min(body,lid),min(knob,min(spout,handle)));
}
fn scene(p:vec3f)->vec2f {let d=object(p);if(p.y+.68<d){return vec2f(p.y+.68,1.);}return vec2f(d,0.);}
fn normal(p:vec3f)->vec3f {let e=.001;return normalize(vec3f(scene(p+vec3f(e,0,0)).x-scene(p-vec3f(e,0,0)).x,scene(p+vec3f(0,e,0)).x-scene(p-vec3f(0,e,0)).x,scene(p+vec3f(0,0,e)).x-scene(p-vec3f(0,0,e)).x));}
fn environment(d:vec3f,blur:f32)->vec3f {
 let sky=mix(vec3f(.05,.07,.1),vec3f(.45,.6,.75),smoothstep(-.2,.7,d.y));
 let softbox=pow(max(dot(d,normalize(vec3f(-.6,1.,.5))),0.),mix(100.,3.,blur))*vec3f(6.,5.7,5.1);
 let strip=pow(max(dot(d,normalize(vec3f(.8,.4,-.6))),0.),mix(180.,5.,blur))*vec3f(1.2,2.,3.);
 return sky+softbox+strip;
}
fn shadow(p:vec3f,l:vec3f)->f32 {var t=.015;var s=1.;for(var i=0;i<28;i++){let d=object(p+l*t);s=min(s,12.*d/t);t+=clamp(d,.025,.2);if(t>4.){break;}}return clamp(s,.08,1.);}
@fragment fn fs(@builtin(position) px:vec4f)->@location(0) vec4f {
 let uv=(px.xy/u.size.xy*2.-1.)*vec2f(u.size.x/u.size.y,-1.);
 let eye=vec3f(sin(u.camera.x)*cos(u.camera.y),sin(u.camera.y),cos(u.camera.x)*cos(u.camera.y))*u.camera.z;
 let f=normalize(vec3f(0,.05,0)-eye);let right=normalize(cross(f,vec3f(0,1,0)));let up=cross(right,f);
 let rd=normalize(f*2.1+right*uv.x+up*uv.y);var t=.0;var hit=vec2f(0.);var found=false;
 for(var i=0;i<100;i++){hit=scene(eye+rd*t);if(hit.x<.001){found=true;break;}t+=hit.x*.75;if(t>16.){break;}}
 var col=environment(rd,.25)*.35;
 if(found){let p=eye+rd*t;let n=normal(p);let v=-rd;let l=normalize(vec3f(sin(u.material.z)*2.,2.4,cos(u.material.z)*2.));let h=normalize(v+l);
 let nl=max(dot(n,l),.001);let nv=max(dot(n,v),.001);let nh=max(dot(n,h),.001);let vh=max(dot(v,h),.001);
 var rough=max(u.material.x,.045);var metal=u.material.y;
 var base=.55+.4*cos(vec3f(0,2.1,4.2)+u.material.w);
 if(hit.y>.5){let checker=(i32(floor(p.x*2.))+i32(floor(p.z*2.)))&1;base=mix(vec3f(.17,.21,.23),vec3f(.25,.29,.3),f32(checker));rough=.7;metal=0.;}
 let f0=mix(vec3f(.04),base,metal);let fres=f0+(1.-f0)*pow(1.-vh,5.);let a=rough*rough;let a2=a*a;
 let denom=nh*nh*(a2-1.)+1.;let D=a2/(3.14159*denom*denom);let k=(rough+1.)*(rough+1.)/8.;let G=nv/(nv*(1.-k)+k)*nl/(nl*(1.-k)+k);
 let spec=D*G*fres/(4.*nv*nl);let diffuse=(1.-fres)*(1.-metal)*base/3.14159;
 let reflected=reflect(-v,n);let envF=f0+(1.-f0)*pow(1.-nv,5.);
 col=(diffuse+spec)*nl*vec3f(4.,3.7,3.3)*shadow(p+n*.01,l);
 col+=environment(reflected,rough*rough)*envF*(1.-rough*.55)+base*(1.-metal)*.09;
 col=mix(environment(rd,.25)*.35,col,exp(-t*.012));}
 col=col/(col+1.);col=pow(max(col,vec3f(0)),vec3f(1./2.2));return vec4f(col,1.);
}`;

async function start() {
  if (!navigator.gpu) throw new Error('浏览器未提供 WebGPU（需要支持它的浏览器与 HTTPS/localhost）');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('当前设备没有可用的 WebGPU adapter');
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: shader });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(m => m.type === 'error');
  if (errors.length) throw new Error(errors.map(e => e.message).join('; '));
  const pipeline = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
  const validation = await device.popErrorScope();
  if (validation) throw new Error(validation.message);
  const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  let ready = true;
  let visible = true;
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; if (visible) draw(); }).observe(canvas);
  device.lost.then(info => { ready = false; fallback(`WebGPU 设备连接中断：${info.reason}`); });
  device.addEventListener('uncapturederror', e => { ready = false; fallback(`WebGPU 绘制错误：${e.error.message}`); });
  function draw() {
    if (!ready || !visible || document.hidden) return;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(devicePixelRatio, 1.5, 1000 / Math.max(rect.width, 1));
    const w = Math.max(1, Math.round(rect.width * scale));
    const h = Math.max(1, Math.round(rect.height * scale));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const values = controls.map(id => Number(document.querySelector('#' + id).value));
    device.queue.writeBuffer(buffer, 0, new Float32Array([w, h, 0, 0, state.yaw, state.pitch, state.distance, 0, ...values, state.scene, 0, 0, 0]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: .02, g: .03, b: .04, a: 1 } }] });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
    device.queue.submit([encoder.finish()]);
  }
  controls.forEach(id => document.querySelector('#' + id).addEventListener('input', () => { document.querySelector('#' + id + '-value').value = Number(document.querySelector('#' + id).value).toFixed(2); draw(); }));
  canvas.addEventListener('pointerdown', e => { state.dragging = true; state.x = e.clientX; state.y = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => { if (!state.dragging) return; state.yaw -= (e.clientX - state.x) * .008; state.pitch = Math.max(-.08, Math.min(1.2, state.pitch + (e.clientY - state.y) * .008)); state.x = e.clientX; state.y = e.clientY; draw(); });
  ['pointerup', 'pointercancel'].forEach(type => canvas.addEventListener(type, () => { state.dragging = false; }));
  canvas.addEventListener('keydown', e => { if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) { e.preventDefault(); state.yaw += (e.key==='ArrowRight'?.12:e.key==='ArrowLeft'?-.12:0); state.pitch=Math.max(-.08,Math.min(1.2,state.pitch+(e.key==='ArrowUp'?.08:e.key==='ArrowDown'?-.08:0)));draw(); } });
  document.querySelector('#preview-shape').addEventListener('change', e => { state.scene = Number(e.target.value); draw(); });
  document.querySelector('#reset-camera').addEventListener('click', () => { Object.assign(state, { yaw: .45, pitch: .28, distance: 4.4 }); draw(); });
  document.querySelectorAll('[data-material-preset]').forEach(button => button.addEventListener('click', () => {
    const preset = button.dataset.materialPreset.split(',');
    controls.slice(0,2).forEach((id,i)=>{const el=document.querySelector('#'+id);el.value=preset[i];el.dispatchEvent(new Event('input'));});
  }));
  new ResizeObserver(draw).observe(canvas);
  document.addEventListener('visibilitychange', draw);
  status.textContent = 'WebGPU 已启用 · 实时 GGX 教学预览 · 非神经网络预测';
  status.dataset.mode = 'webgpu';
  document.querySelector('#gpu-fallback').hidden = true;
  canvas.hidden = false;
  draw();
}
function fallback(reason) {
  canvas.hidden = true;
  document.querySelector('#gpu-fallback').hidden = false;
  status.textContent = `${reason}。以下回退到已测参考图；材质滑块不控制静态图。`;
  status.dataset.mode = 'fallback';
  document.querySelectorAll('#material-controls input,#material-controls select,#material-controls button').forEach(el => el.disabled = true);
}
start().catch(error => fallback(error.message));

