/* HFX3D — Geospatial coverage map (deck.gl) + mini point cloud viewer */
'use strict';

/* ════════════════════════════════════════
   Mini PLY viewer — shared with map panel
   ════════════════════════════════════════ */
(function () {
  if (typeof THREE === 'undefined') return;

  const VERT = `
    attribute vec3 aRgb;
    attribute vec3 aSem;
    uniform int uMode;
    varying vec3 vC;
    void main() {
      if (uMode == 1) vC = aSem; else vC = aRgb;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = 1.5;
      gl_Position  = projectionMatrix * mv;
    }
  `;
  const FRAG = `
    precision mediump float;
    varying vec3 vC;
    void main() { gl_FragColor = vec4(vC, 1.0); }
  `;


  var _mv = {
    renderer: null, scene: null, camera: null, controls: null,
    cloud: null, worker: null, mode: 0, needsRender: false, pendingId: null,
  };

  function init() {
    var canvas = document.getElementById('gpCanvas');
    if (!canvas) return;

    _mv.renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: false });
    _mv.renderer.setPixelRatio(1);
    _mv.scene = new THREE.Scene();
    _mv.scene.background = new THREE.Color(0x0a0a14);

    var w = canvas.clientWidth || 320, h = canvas.clientHeight || 180;
    _mv.renderer.setSize(w, h, false);

    _mv.camera = new THREE.PerspectiveCamera(52, w / h, 0.01, 10000);
    _mv.camera.position.set(0, -45, 22);
    _mv.camera.up.set(0, 0, 1);

    _mv.controls = new THREE.OrbitControls(_mv.camera, canvas);
    _mv.controls.enableDamping = true;
    _mv.controls.dampingFactor = 0.05;
    _mv.controls.autoRotate = true;
    _mv.controls.autoRotateSpeed = 0.7;
    _mv.controls.screenSpacePanning = false;
    _mv.controls.addEventListener('change', function () { _mv.needsRender = true; });

    new ResizeObserver(function () {
      var cw = canvas.clientWidth, ch = canvas.clientHeight;
      if (!cw || !ch) return;
      _mv.renderer.setSize(cw, ch, false);
      _mv.camera.aspect = cw / ch;
      _mv.camera.updateProjectionMatrix();
      _mv.needsRender = true;
    }).observe(canvas);

    try {
      _mv.worker = new Worker('static/js/ply-worker.js?v=2');
    } catch (e) { console.warn('mini viewer: worker unavailable', e); }

    _mv.visible = false;
    new IntersectionObserver(function(entries) {
      _mv.visible = entries[0].isIntersecting;
      if (_mv.visible) _mv.needsRender = true;
    }, { threshold: 0 }).observe(canvas);

    (function loop() {
      requestAnimationFrame(loop);
      if (!_mv.visible) return;
      var moved = _mv.controls && _mv.controls.update();
      if (!_mv.needsRender && !moved) return;
      _mv.needsRender = false;
      if (_mv.cloud) _mv.renderer.render(_mv.scene, _mv.camera);
    })();
  }

  function _ingest(d) {
    if (_mv.cloud) {
      _mv.scene.remove(_mv.cloud);
      _mv.cloud.geometry.dispose();
      _mv.cloud.material.dispose();
      _mv.cloud = null;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(d.xyz, 3));
    geo.setAttribute('aRgb',     new THREE.BufferAttribute(d.rgb, 3));
    geo.setAttribute('aSem',     new THREE.BufferAttribute(d.sem, 3));
    geo.computeBoundingBox();
    var cen = new THREE.Vector3();
    geo.boundingBox.getCenter(cen);
    geo.translate(-cen.x, -cen.y, -cen.z);
    _mv.cloud = new THREE.Points(geo, new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG,
      uniforms: { uMode: { value: _mv.mode } },
    }));
    _mv.scene.add(_mv.cloud);
    var size = new THREE.Box3().setFromObject(_mv.cloud).getSize(new THREE.Vector3()).length();
    _mv.camera.position.set(0, -size * 0.72, size * 0.36);
    _mv.camera.far = size * 25;
    _mv.camera.updateProjectionMatrix();
    _mv.controls.target.set(0, 0, 0);
    _mv.controls.update();
    _mv.controls.saveState();
    _mv.needsRender = true;
  }

  window._miniViewerLoad = function (bldId) {
    if (!_mv.worker) return;
    var loading = document.getElementById('gpLoading');
    var noVideo = document.getElementById('gpNoVideo');
    if (loading) loading.style.display = 'flex';
    if (noVideo) noVideo.style.display = 'none';

    var id = bldId + '_' + Date.now();
    _mv.pendingId = id;
    var url = 'https://hfx3d-ply.zazukhan58.workers.dev/' + bldId + '_preview.ply';

    function handler(e) {
      if (e.data.id !== id) return;
      _mv.worker.removeEventListener('message', handler);
      if (_mv.pendingId !== id) return;
      if (loading) loading.style.display = 'none';
      if (e.data.error) { if (noVideo) { noVideo.style.display = 'flex'; } return; }
      _ingest(e.data);
    }
    _mv.worker.addEventListener('message', handler);
    _mv.worker.postMessage({ url: url, id: id });
  };

  window._miniViewerSetMode = function (mode) {
    _mv.mode = mode === 'semantic' ? 1 : 0;
    if (_mv.cloud) {
      _mv.cloud.material.uniforms.uMode.value = _mv.mode;
      _mv.needsRender = true;
    }
  };

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', init);
  else
    init();
})();


/* ════════════════════════════════════════
   Map (deck.gl + Mapbox)
   ════════════════════════════════════════ */
(function () {
  if (typeof deck === 'undefined' || typeof mapboxgl === 'undefined') return;
  if (!document.getElementById('coverageMap')) return;

  const TOKEN = 'pk.eyJ1IjoiYWJkaTEyMyIsImEiOiJjbXE3Y2d0czEwNXM1MnVxNWpoOWdlaGc4In0.4Bhdi3xKXTnopmel4diuKQ';

  /* ── Canonical semantic class colours ── */
  const SEM_CLASSES = [
    { name: 'Wall',       hex: '#FF2828' },
    { name: 'Window',     hex: '#00CCFF' },
    { name: 'Door',       hex: '#005449' },
    { name: 'Balcony',    hex: '#FFEE00' },
    { name: 'Vegetation', hex: '#00FF50' },
    { name: 'Stairs',     hex: '#FF99FF' },
    { name: 'Terrain',    hex: '#B84600' },
    { name: 'Roof',       hex: '#FF00B4' },
    { name: 'Blinds',     hex: '#7800FF' },
    { name: 'Other',      hex: '#A0A0A0' },
    { name: 'Column',     hex: '#4F0004' },
    { name: 'Arch',       hex: '#000000' },
  ];

  /* ── Helpers ── */
  function pinColor(cls) {
    return cls >= 11 ? '#f97316' : cls >= 10 ? '#f59e0b' : cls >= 9 ? '#8b5cf6' : cls >= 8 ? '#6d28d9' : '#4338ca';
  }
  function hex2rgb(hex, a) {
    return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16), a ?? 220];
  }
  function ease(t) { return 1 - Math.pow(1 - t, 3); }

  /* ── UI refs ── */
  const _elId    = document.getElementById('gpId');
  const _elName  = document.getElementById('gpName');
  const _elUni   = document.getElementById('gpUni');
  const _elPts   = document.getElementById('gpPts');
  const _elCls   = document.getElementById('gpCls');
  const _elCta   = document.getElementById('gpCta');
  const _elModes = document.getElementById('gpModes');
  const _elLegend = document.getElementById('gpLegend');

  /* ── Legend ── */
  function _buildLegend() {
    if (!_elLegend) return;
    _elLegend.innerHTML = SEM_CLASSES.map(c =>
      `<div class="gp-legend-item">` +
      `<span class="gp-lc" style="background:${c.hex}"></span>${c.name}` +
      `</div>`
    ).join('');
  }

  /* ── Mode toggle ── */
  let _currentBld = null;

  document.querySelectorAll('.gp-mode').forEach(btn =>
    btn.addEventListener('click', () => {
      document.querySelectorAll('.gp-mode').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (_elLegend) _elLegend.style.opacity = btn.dataset.mode === 'semantic' ? '1' : '0.4';
      if (typeof window._miniViewerSetMode === 'function')
        window._miniViewerSetMode(btn.dataset.mode);
    })
  );

  /* ── Pin atlas ── */
  let _pinAtlas;

  function createPinAtlas() {
    const c = document.createElement('canvas');
    c.width = 32; c.height = 44;
    const ctx = c.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(16, 14, 13, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(9, 23); ctx.lineTo(23, 23); ctx.lineTo(16, 43);
    ctx.closePath();
    ctx.fill();
    return c;
  }

  /* ── deck.gl ── */
  let _overlay, _map;
  let _animT = 0, _animating = false, _selectedId = null, _buildings = [];

  function buildLayers(t) {
    const vis = _buildings.filter(b => b.lat !== null);
    return [
      new deck.ScatterplotLayer({
        id: 'dots', data: vis,
        getPosition: d => [d.lon, d.lat],
        getRadius: 8, radiusUnits: 'meters',
        getFillColor: d => hex2rgb(pinColor(d.cls), Math.round(130 * Math.min(1, t * 3))),
        stroked: false,
        updateTriggers: { getFillColor: [t] },
      }),
      new deck.IconLayer({
        id: 'pins', data: vis,
        iconAtlas: _pinAtlas,
        iconMapping: { pin: { x: 0, y: 0, width: 32, height: 44, anchorY: 44, mask: true } },
        getIcon: () => 'pin',
        getPosition: d => [d.lon, d.lat],
        getSize: d => (_selectedId === d.id ? 40 : 32) *
                      ease(Math.max(0, Math.min(1, (t - d._delay) / 0.75))),
        getColor: d => _selectedId === d.id ? [255, 255, 255, 255] : hex2rgb(pinColor(d.cls)),
        sizeUnits: 'pixels',
        pickable: true,
        autoHighlight: true, highlightColor: [255, 255, 255, 80],
        updateTriggers: {
          getSize:  [t, _selectedId],
          getColor: [_selectedId],
        },
        onClick: ({ object }) => { if (object) selectBuilding(object); },
      }),
    ];
  }

  function startAnim() {
    if (_animating) return;
    _animating = true;
    const dur = 2400, start = performance.now();
    (function frame(now) {
      _animT = Math.min((now - start) / dur, 1);
      if (_overlay) _overlay.setProps({ layers: buildLayers(_animT) });
      if (_animT < 1) requestAnimationFrame(frame);
    })(performance.now());
  }

  function refreshLayers() {
    if (_overlay) _overlay.setProps({ layers: buildLayers(_animT) });
  }

  /* ── Building selection ── */
  let _firstLoad = true;

  function selectBuilding(bld) {
    _selectedId  = bld.id;
    _currentBld  = bld;
    refreshLayers();

    _elId.textContent   = bld.id;
    _elName.textContent = bld.name;
    _elUni.textContent  = bld.uni;
    _elPts.textContent  = bld.pts;
    _elCls.textContent  = bld.cls;
    _elCta.onclick = () =>
      typeof window.loadBuildingViewer === 'function' && window.loadBuildingViewer(bld.id);

    if (bld.lat && !_firstLoad && _map) {
      _map.flyTo({ center: [bld.lon, bld.lat], zoom: 17, pitch: 55, bearing: -15, duration: 1400 });
    }
    _firstLoad = false;

    if (typeof window._miniViewerLoad === 'function') window._miniViewerLoad(bld.id);
  }

  /* ── Map + overlay init ── */
  function initMap(buildings) {
    _buildings = buildings;
    buildings.filter(b => b.lat !== null).forEach((b, i, a) => { b._delay = (i / a.length) * 0.55; });

    mapboxgl.accessToken = TOKEN;
    _map = new mapboxgl.Map({
      container: 'coverageMap',
      style: 'mapbox://styles/mapbox/standard',
      center: [-63.574, 44.648],
      zoom: 12.3, pitch: 45, bearing: -15,
      antialias: true,
    });
    _map.addControl(new mapboxgl.NavigationControl({ showCompass: true }), 'top-left');

    _overlay = new deck.MapboxOverlay({
      interleaved: false,
      layers: buildLayers(0),
      getTooltip: ({ object }) => object && {
        html: `<div class="geo-tip-id">${object.id}</div>
               <div class="geo-tip-name">${object.name}</div>
               <div class="geo-tip-stats">${object.pts}&nbsp;pts &middot; ${object.cls}&nbsp;classes</div>`,
        style: {
          background: 'rgba(13,13,24,0.92)', color: '#f1f5f9',
          borderRadius: '8px', padding: '8px 12px',
          fontSize: '12px', lineHeight: '1.55',
          border: '1px solid rgba(255,255,255,0.12)',
          backdropFilter: 'blur(10px)',
          fontFamily: 'Inter, sans-serif', pointerEvents: 'none',
        },
      },
    });
    _map.addControl(_overlay);
    window.addEventListener('resize', () => _map.resize());
  }

  /* ── Boot — lazy: Mapbox only created when section enters view ── */
  fetch('assets/data/buildings.json')
    .then(r => r.json())
    .then(buildings => {
      _pinAtlas = createPinAtlas();
      _buildLegend();

      const sec = document.getElementById('geospatial');
      if (!sec) return;
      let _inited = false;
      new IntersectionObserver(([e]) => {
        if (!e.isIntersecting) return;
        if (!_inited) {
          _inited = true;
          initMap(buildings);
          const def = buildings.find(b => b.id === 'BLD009') || buildings.find(b => b.lat !== null);
          if (def) selectBuilding(def);
        }
        startAnim();
      }, { threshold: 0.05 }).observe(sec);
    });
})();
