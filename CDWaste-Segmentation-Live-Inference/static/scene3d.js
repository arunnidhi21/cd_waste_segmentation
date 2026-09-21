(function () {
  var container = document.getElementById("scene3d");
  if (!container || typeof THREE === "undefined") return;

  var LIME = 0xd8ff73;

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 15);

  var renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    return; // no WebGL support — fail quietly, page still works without the 3D layer
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0x333333, 1.3));
  var key = new THREE.DirectionalLight(LIME, 1.7);
  key.position.set(6, 7, 9);
  scene.add(key);
  var rim = new THREE.DirectionalLight(0x4488ff, 0.35);
  rim.position.set(-6, -4, -5);
  scene.add(rim);

  // One fragment per CDWaste material class, each a distinct low-poly form
  function makeFragment(geometry, color, metalness, roughness) {
    var group = new THREE.Group();
    var mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({ color: color, metalness: metalness, roughness: roughness, flatShading: true })
    );
    group.add(mesh);
    var edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: LIME, transparent: true, opacity: 0.22 })
    );
    group.add(edges);
    return group;
  }

  var fragments = [
    { mesh: makeFragment(new THREE.BoxGeometry(2.3, 2.3, 2.3), 0x1c1c18, 0.05, 0.95), pos: [-5.2, 2.3, -3] },      // concrete
    { mesh: makeFragment(new THREE.BoxGeometry(3.1, 1.1, 1.5), 0x5c2a1e, 0.05, 0.85), pos: [4.8, -1.8, -5] },      // brick
    { mesh: makeFragment(new THREE.CylinderGeometry(0.9, 0.9, 2.5, 8), 0x4a3524, 0.05, 0.8), pos: [-3.8, -3.2, -6] }, // wood
    { mesh: makeFragment(new THREE.IcosahedronGeometry(1.4, 0), 0x9a9a92, 0.9, 0.2), pos: [5.4, 3.2, -6.5] },      // metal
    { mesh: makeFragment(new THREE.SphereGeometry(1.2, 12, 10), 0x161616, 0.15, 0.1), pos: [0.4, -4.6, -3.5] }     // plastic
  ];

  fragments.forEach(function (f, i) {
    f.mesh.position.set(f.pos[0], f.pos[1], f.pos[2]);
    f.mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * 0.4);
    f.spin = { x: 0.0018 + i * 0.0004, y: 0.0026 + i * 0.0003 };
    f.bobSeed = i;
    scene.add(f.mesh);
  });

  var mouseX = 0, mouseY = 0;
  window.addEventListener("mousemove", function (e) {
    mouseX = e.clientX / window.innerWidth - 0.5;
    mouseY = e.clientY / window.innerHeight - 0.5;
  });

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }
  window.addEventListener("resize", onResize);

  var clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    var t = clock.getElapsedTime();
    fragments.forEach(function (f) {
      f.mesh.rotation.x += f.spin.x;
      f.mesh.rotation.y += f.spin.y;
      f.mesh.position.y += Math.sin(t * 0.6 + f.bobSeed) * 0.0015;
    });
    camera.position.x += (mouseX * 3 - camera.position.x) * 0.02;
    camera.position.y += (-mouseY * 2 - camera.position.y) * 0.02;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  animate();
  requestAnimationFrame(function () { container.classList.add("ready"); });
})();
