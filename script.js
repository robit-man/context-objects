/*  Three.js + viewer logic
    ─────────────────────────────────────────────────────────────── */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/controls/OrbitControls.js';

/* ---------- DOM Short‑cuts ---------- */
const selPane = document.getElementById('selectednode');
const ageSel  = document.getElementById('ageFilter');
const typeSel = document.getElementById('typeFilter');

/* ---------- Scene / Camera ---------- */
const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera   = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 2000);
camera.position.set(0, 50, 180);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

/* ---------- Raycaster ---------- */
const raycaster = new THREE.Raycaster();
const pointer   = new THREE.Vector2();

/* ---------- Storage ---------- */
const nodeMap  = new Map();          // context_id → { data, mesh }
const levelMap = new Map();          // level (int)  → array of ids
const edges    = [];                 // THREE.Line objects

/* ---------- Helpers ---------- */
function parseTimestamp(ts){
  /* "20250703T195022Z" → Date */
  const y = +ts.slice(0,4);
  const m = +ts.slice(4,6)-1;
  const d = +ts.slice(6,8);
  const hh= +ts.slice(9,11);
  const mm= +ts.slice(11,13);
  const ss= +ts.slice(13,15);
  return new Date(Date.UTC(y,m,d,hh,mm,ss));
}

function unixNow(){ return Date.now(); }

/* ---------- Load & Build Graph ---------- */
fetch('context.jsonl')
  .then(r => r.text())
  .then(text => {
      const objects = text.trim().split(/\r?\n/).map(JSON.parse);
      if(!objects.length){ throw new Error('context.jsonl is empty or invalid'); }

      /* — root = most recent timestamp — */
      objects.sort((a,b) => parseTimestamp(b.timestamp) - parseTimestamp(a.timestamp));
      const rootObj = objects[0];
      const rootId  = rootObj.context_id;

      /* — index all objects — */
      objects.forEach(o => nodeMap.set(o.context_id, {data:o, mesh:null, pos:new THREE.Vector3(), level:-1}));

      /* — BFS walk along `.references` outward to assign levels — */
      const queue = [rootId];
      nodeMap.get(rootId).level = 0;
      while(queue.length){
          const id = queue.shift();
          const node = nodeMap.get(id);
          const lev  = node.level;
          (node.data.references||[]).forEach(refId => {
              if(!nodeMap.has(refId)) return;
              const ref = nodeMap.get(refId);
              if(ref.level === -1 || ref.level > lev+1){
                  ref.level = lev+1;
                  queue.push(refId);
              }
          });
      }

      /* Node placement */
      const levelHeight = 25, radiusStep = 30;
      nodeMap.forEach((obj, id)=>{
          const lv = obj.level<0 ? 0 : obj.level;                // unknown → root level
          const peers = (levelMap.get(lv) || []);
          peers.push(id);
          levelMap.set(lv, peers);
      });

      /* Generate positions radially per level */
      levelMap.forEach((ids, lv)=>{
          const radius = lv * radiusStep;
          ids.forEach((cid, idx)=>{
              const angle = (idx/ids.length) * Math.PI*2;
              const x = radius * Math.cos(angle);
              const z = radius * Math.sin(angle);
              const y = lv * levelHeight;
              nodeMap.get(cid).pos.set(x,y,z);
          });
      });

      /* Add meshes */
      const colorByDomain = {
          segment:0x2ca0ff, stage:0xff7850, artifact:0x8bc34a
      };
      const sphereGeo = new THREE.SphereGeometry(5, 24, 24);
      nodeMap.forEach((o, id)=>{
          const dom  = o.data.domain || 'artifact';
          const col  = colorByDomain[dom] ?? 0xaaaaaa;
          const mat  = new THREE.MeshStandardMaterial({color:col});
          const mesh = new THREE.Mesh(sphereGeo, mat);
          mesh.position.copy(o.pos);
          mesh.userData = {contextId:id};
          scene.add(mesh);
          o.mesh = mesh;
      });

      /* Edges (references only) */
      const lineMat = new THREE.LineBasicMaterial({color:0x666666});
      nodeMap.forEach((o, id)=>{
          o.data.references.forEach(refId=>{
              const to = nodeMap.get(refId);
              if(!to) return;
              const g = new THREE.BufferGeometry().setFromPoints([o.pos, to.pos]);
              const line = new THREE.Line(g, lineMat.clone());
              scene.add(line);
              edges.push(line);
          });
      });

      /* Lighting */
      const amb = new THREE.AmbientLight(0xffffff, 0.55);
      scene.add(amb);
      const dir = new THREE.DirectionalLight(0xffffff, 0.8);
      dir.position.set(60,120,80);
      scene.add(dir);

      /* UI filters setup */
      populateTypeFilter(objects);
      ageSel.addEventListener('change', applyFilters);
      typeSel.addEventListener('change', applyFilters);

      /* Kick‑off render loop */
      animate();
  })
  .catch(err => { selPane.textContent = 'Error: '+err.message; console.error(err); });

function populateTypeFilter(objs){
    /* collect unique domains, components & tags */
    const bucket = new Set();
    objs.forEach(o=>{
        bucket.add('domain:'+o.domain);
        bucket.add('component:'+o.component);
        (o.tags||[]).forEach(t => bucket.add('tag:'+t));
    });
    [...bucket].sort().forEach(key=>{
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = key;
        typeSel.appendChild(opt);
    });
}

function applyFilters(){
    const ageVal  = ageSel.value;      // "all", "h1", "d1", "w1", "m1"
    const typeVal = typeSel.value;     // "all" or "domain:segment", "tag:prompt", …

    const now = unixNow();
    const maxAgeMs = {
        h1:3600e3, d1:86400e3, w1:86400e3*7, m1:86400e3*30
    }[ageVal];

    nodeMap.forEach((o)=>{
        let visible = true;

        /* age filter */
        if(maxAgeMs){
            const objAge = now - parseTimestamp(o.data.timestamp).getTime();
            visible = visible && (objAge <= maxAgeMs);
        }

        /* type / tag filter */
        if(typeVal !== 'all'){
            const [kind,val] = typeVal.split(':',2);
            if(kind==='domain')      visible = visible && o.data.domain===val;
            else if(kind==='component') visible = visible && o.data.component===val;
            else if(kind==='tag')    visible = visible && (o.data.tags||[]).includes(val);
        }

        o.mesh.visible = visible;
    });

    /* hide/show edges if either endpoint hidden */
    edges.forEach(ln=>{
        const pts = ln.geometry.attributes.position.array;
        const a   = new THREE.Vector3(pts[0],pts[1],pts[2]);
        const b   = new THREE.Vector3(pts[3],pts[4],pts[5]);
        let m1,m2;
        nodeMap.forEach(n=>{
            if(!m1 && n.pos.equals(a)) m1=n.mesh;
            if(!m2 && n.pos.equals(b)) m2=n.mesh;
        });
        ln.visible = (m1?.visible && m2?.visible);
    });
}

/* ---------- Interaction ---------- */
function onPointerMove(ev){
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x =  (ev.clientX - rect.left) / rect.width  *  2 - 1;
    pointer.y = -(ev.clientY - rect.top ) / rect.height *  2 + 1;
}
renderer.domElement.addEventListener('pointermove', onPointerMove);
renderer.domElement.addEventListener('click', onPointerMove);

/* ---------- Render Loop ---------- */
function animate(){
    requestAnimationFrame(animate);
    controls.update();

    /* pick */
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects([...nodeMap.values()].map(n=>n.mesh).filter(m=>m.visible));
    if(intersects[0]){
        const id = intersects[0].object.userData.contextId;
        const obj = nodeMap.get(id).data;
        selPane.textContent = JSON.stringify(obj, null, 2);
    }

    renderer.render(scene, camera);
}

/* ---------- Resize ---------- */
addEventListener('resize', ()=>{
    camera.aspect = innerWidth/innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});
