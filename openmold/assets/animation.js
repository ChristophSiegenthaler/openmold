// 3D fill animation. Expects window.FA = {parts:[{id,label,colour}], tmax, views:{name:{p,t}}}, mesh data in <script id="mesh-ID">.
(function(){
const FA=window.FA, TMAX=FA.tmax;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const grid=document.getElementById("fa-grid"), tIn=document.getElementById("fa-t"), tRead=document.getElementById("fa-tread"), playB=document.getElementById("fa-play");
const camera=new THREE.PerspectiveCamera(28,4/3,0.1,20000), target=new THREE.Vector3();
let T=TMAX*0.6, playing=false, speed=0.25, last=0, showMarks=true, syncing=false, hold=0, pend=false;
const vps=[];
const vert=`attribute float tf; varying float vTf; varying vec3 vPos;
void main(){vTf=tf; vec4 mv=modelViewMatrix*vec4(position,1.0); vPos=mv.xyz; gl_Position=projectionMatrix*mv;}`;
const frag=`uniform float uT; uniform float uMax; varying float vTf; varying vec3 vPos;
vec3 jet(float x){return clamp(vec3(1.5-abs(4.0*x-3.0),1.5-abs(4.0*x-2.0),1.5-abs(4.0*x-1.0)),0.0,1.0);}
void main(){ if(vTf>uT) discard;
  vec3 n=normalize(cross(dFdx(vPos),dFdy(vPos)));
  vec3 c=jet(vTf/uMax)*(0.42+0.58*abs(n.z));
  float band=smoothstep(0.015*uMax,0.0,uT-vTf)*step(uT,uMax*0.999);
  gl_FragColor=vec4(mix(c,vec3(1.0),0.85*band),1.0);}`;
function makeVP(part){
  const el=document.createElement("div"); el.className="vp"; el.style.setProperty("--k",part.colour);
  el.innerHTML='<canvas></canvas><div class="cap"><span class="tag"></span><span class="d"></span></div><div class="load">Loading mesh…</div>';
  el.querySelector(".tag").textContent=part.id; el.querySelector(".d").textContent=part.label;
  grid.appendChild(el);
  const canvas=el.querySelector("canvas"); canvas.setAttribute("aria-label","3D fill view of "+part.id);
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  const controls=new THREE.OrbitControls(camera,canvas); controls.target=target; controls.enableZoom=false;
  controls.addEventListener("change",()=>{ if(syncing) return; syncing=true; vps.forEach(v=>v.controls!==controls&&v.controls.update()); syncing=false; renderAll();});
  const v={part,el,canvas,renderer,scene:new THREE.Scene(),controls,mat:null}; vps.push(v);
}
function load(v){
  const bin=atob(document.getElementById("mesh-"+v.part.id).textContent.trim());
  const u8=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
  const buf=u8.buffer, [nv,nt,nw,na,wide]=new Uint32Array(buf,0,5), scale=new Float32Array(buf,20,1)[0]; let o=24;
  const qp=new Int16Array(buf,o,nv*3); o+=nv*6; const qt=new Uint16Array(buf,o,nv); o+=nv*2;
  const idx=wide?new Uint32Array(buf,o,nt*3):new Uint16Array(buf,o,nt*3); o+=nt*3*(wide?4:2); o+=(4-o%4)%4;
  const W=new Float32Array(buf,o,nw*4); o+=nw*16; const A=new Float32Array(buf,o,na*4);
  const pos=new Float32Array(nv*3); for(let i=0;i<pos.length;i++) pos[i]=qp[i]*scale;
  const tf=new Float32Array(nv); for(let i=0;i<nv;i++) tf[i]=qt[i]/65535*TMAX;
  const g=new THREE.BufferGeometry(); g.setAttribute("position",new THREE.BufferAttribute(pos,3)); g.setAttribute("tf",new THREE.BufferAttribute(tf,1)); g.setIndex(new THREE.BufferAttribute(idx,1));
  v.mat=new THREE.ShaderMaterial({vertexShader:vert,fragmentShader:frag,uniforms:{uT:{value:T},uMax:{value:TMAX}},side:THREE.DoubleSide,extensions:{derivatives:true}});
  v.scene.add(new THREE.Mesh(g,v.mat));
  v.ghost=new THREE.Mesh(g,new THREE.MeshBasicMaterial({transparent:true,opacity:0.07,depthWrite:false,side:THREE.DoubleSide}));
  v.edges=new THREE.LineSegments(new THREE.EdgesGeometry(g,28),new THREE.LineBasicMaterial({transparent:true,opacity:0.45}));
  v.scene.add(v.ghost,v.edges);
  const pts=(arr,n)=>{const p=new Float32Array(n*3),t=new Float32Array(n); for(let i=0;i<n;i++){p.set(arr.subarray(i*4,i*4+3),i*3); t[i]=arr[i*4+3];}
    const gg=new THREE.BufferGeometry(); gg.setAttribute("position",new THREE.BufferAttribute(p,3));
    const m=new THREE.Points(gg,new THREE.PointsMaterial({size:4.5,sizeAttenuation:false,depthTest:false})); m.renderOrder=3; m.userData.t=t; return m;};
  v.weld=pts(W,nw); v.air=pts(A,na); v.scene.add(v.weld,v.air);
  v.el.querySelector(".load").remove();
}
function countBelow(t,arr){let n=0; while(n<arr.length&&arr[n]<=t) n++; return n;}
function setT(t){T=Math.max(0,Math.min(TMAX,t)); tIn.value=T; tRead.innerHTML=T.toFixed(2)+" <small>s</small>";
  vps.forEach(v=>{ if(!v.mat) return; v.mat.uniforms.uT.value=T;
    v.weld.geometry.setDrawRange(0,showMarks?countBelow(T,v.weld.userData.t):0);
    v.air.geometry.setDrawRange(0,showMarks?countBelow(T,v.air.userData.t):0);}); renderAll();}
function size(){vps.forEach(v=>{const w=v.canvas.clientWidth,h=v.canvas.clientHeight; if(w&&h){v.renderer.setSize(w,h,false); camera.aspect=w/h;}}); camera.updateProjectionMatrix(); renderAll();}
function renderAll(){ if(pend) return; pend=true; requestAnimationFrame(()=>{pend=false; vps.forEach(v=>v.renderer.render(v.scene,camera));}); }
function theme(){const bg=new THREE.Color(css("--panel")), gh=new THREE.Color(css("--ghost"));
  vps.forEach(v=>{v.renderer.setClearColor(bg); if(!v.mat) return; v.ghost.material.color=gh; v.edges.material.color=gh;
    v.weld.material.color=new THREE.Color(css("--weld")); v.air.material.color=new THREE.Color(css("--air"));}); renderAll();}
function view(name){const s=FA.views[name]; camera.up.set(...FA.up); camera.position.set(...s.p); target.set(...s.t); camera.lookAt(target); syncing=true; vps.forEach(v=>v.controls.update()); syncing=false; renderAll();
  document.querySelectorAll("#fa-views button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===name));}
function tick(ts){ if(!playing) return; const dt=last?(ts-last)/1000:0; last=ts;
  if(T>=TMAX){hold+=dt; if(hold>1.5){hold=0; setT(0);}} else setT(T+dt*speed*TMAX/2);
  requestAnimationFrame(tick);}
function setPlay(on){playing=on; playB.textContent=on?"Pause":"Play"; if(on){ if(T>=TMAX) setT(0); last=0; requestAnimationFrame(tick);} }
playB.onclick=()=>setPlay(!playing);
tIn.oninput=()=>{setPlay(false); setT(+tIn.value);};
document.getElementById("fa-speed").onclick=e=>{const b=e.target.closest("button"); if(!b) return; speed=+b.dataset.s;
  document.querySelectorAll("#fa-speed button").forEach(x=>x.setAttribute("aria-pressed",x===b));};
document.getElementById("fa-views").onclick=e=>{const b=e.target.closest("button"); if(b) view(b.dataset.v);};
document.getElementById("fa-marks").onchange=e=>{showMarks=e.target.checked; setT(T);};
document.getElementById("fa-zoom").onclick=e=>{const b=e.target.closest("button"); if(!b) return; camera.position.sub(target).multiplyScalar(+b.dataset.z).add(target); renderAll();};
new ResizeObserver(size).observe(grid);
new IntersectionObserver(es=>{ if(!es[0].isIntersecting&&playing) setPlay(false); }).observe(grid);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",theme);
new MutationObserver(theme).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
tIn.max=TMAX; tIn.step=TMAX/400;
FA.parts.forEach(makeVP); view("gate"); size();
vps.forEach(v=>{ try{ load(v); }catch(err){ const l=v.el.querySelector(".load"); if(l) l.textContent="Could not show this part: "+err.message; } });
theme(); setT(T);
})();
