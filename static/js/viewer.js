/* HFX3D — Interactive point cloud viewer
   4 perf optimisations vs. previous version:
   1. Single WebGLRenderer, dual viewport via setScissor/setViewport
   2. Render-on-demand (idle = 0 GPU work)
   3. GLSL ShaderMaterial — all 4 colour arrays as vertex attrs, mode = uniform (no buffer re-upload on switch)
   4. Inline Web Worker — fetch + parse + colourmap off the main thread
*/
'use strict';

(function () {
  if (typeof THREE === 'undefined') { console.warn('HFX viewer: Three.js not loaded.'); return; }

  /* ── Canonical class colours (must match pre-rendered videos & map legend) ── */
  const CLS_COLORS = [
    [1.000,0.157,0.157],[0.000,0.800,1.000],[0.000,0.329,0.286],
    [1.000,0.933,0.000],[0.000,1.000,0.314],[1.000,0.600,1.000],
    [0.722,0.275,0.000],[1.000,0.000,0.706],[0.471,0.000,1.000],
    [0.627,0.627,0.627],[0.310,0.000,0.016],[0.000,0.000,0.000],
  ];
  const CLS_NAMES = ['Wall','Window','Door','Balcony','Vegetation',
                     'Stairs','Terrain','Roof','Blinds','Other','Column','Arch'];
  const CLS_HEX   = ['#FF2828','#00CCFF','#005449','#FFEE00','#00FF50',
                     '#FF99FF','#B84600','#FF00B4','#7800FF','#A0A0A0','#4F0004','#000000'];

  /* ── Building metadata ── */
  const BUILDINGS = [
    { id:'BLD009', name:'Halifax Central Library', pts:'23.1M', cls:8  },
    { id:'BLD001', name:'Goldberg',                pts:'19.4M', cls:7  },
    { id:'BLD002', name:'Mona Campbell A',         pts:'4.8M',  cls:7  },
    { id:'BLD003', name:'Mona Campbell B',         pts:'7.8M',  cls:9  },
    { id:'BLD004', name:'Kenneth Rowe',            pts:'16.5M', cls:10 },
    { id:'BLD005', name:'Schulich Law',            pts:'47.9M', cls:9  },
    { id:'BLD006', name:'VIC Suites',              pts:'14.1M', cls:12 },
    { id:'BLD007', name:'Engineering Bldg',        pts:'13.9M', cls:8  },
    { id:'BLD008', name:'Mona Campbell C',         pts:'7.9M',  cls:10 },
    { id:'BLD010', name:'Emera Place',             pts:'32.7M', cls:10 },
    { id:'BLD011', name:'Dal Gerard Hall',         pts:'16.5M', cls:9  },
    { id:'BLD012', name:'The Margarita',           pts:'31.9M', cls:10 },
    { id:'BLD013', name:'1801 Hollis St #1',       pts:'38.3M', cls:11 },
    { id:'BLD014', name:'Summit Place',            pts:'27.5M', cls:9  },
    { id:'BLD015', name:'1801 Hollis St #2',       pts:'1.9M',  cls:8  },
    { id:'BLD016', name:'1801 Hollis St #3',       pts:'5.0M',  cls:9  },
    { id:'BLD017', name:'1801 Hollis St #4',       pts:'10.1M', cls:9  },
    { id:'BLD018', name:'The Exchange',             pts:'16.3M', cls:9  },
    { id:'BLD019', name:'Sexton Campus 1',         pts:'—',     cls:9  },
    { id:'BLD020', name:'Sexton Campus 2',         pts:'—',     cls:9  },
    { id:'BLD021', name:'TD Building',             pts:'—',     cls:10 },
    { id:'BLD022', name:'Steele Ocean 1',          pts:'26.2M', cls:10 },
    { id:'BLD023', name:'Steele Ocean 2',          pts:'—',     cls:9  },
    { id:'BLD024', name:'Sobeys Hub',              pts:'26.9M', cls:11 },
    { id:'BLD025', name:'Belmont House',           pts:'—',     cls:9  },
    { id:'BLD026', name:'The Avery',               pts:'—',     cls:9  },
  ];

  const MODE_IDX = { rgb:0, semantic:1, intensity:2, time:3 };

  /* ════════════════════════════════════════════
     GLSL — vertex shader selects colour array
     via uniform int uMode; no data re-upload
     on mode switch.
     ════════════════════════════════════════════ */
  const VERT_SRC = `
    attribute vec3 aRgb;
    attribute vec3 aSem;
    attribute vec3 aInt;
    attribute vec3 aTim;
    uniform int uMode;
    varying vec3 vC;
    void main() {
      if      (uMode == 1) vC = aSem;
      else if (uMode == 2) vC = aInt;
      else if (uMode == 3) vC = aTim;
      else                 vC = aRgb;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = 1.5;
      gl_Position  = projectionMatrix * mv;
    }
  `;
  const FRAG_SRC = `
    precision mediump float;
    varying vec3 vC;
    void main() { gl_FragColor = vec4(vC, 1.0); }
  `;

  /* ════════════════════════════════════════════
     Inline Web Worker — fetch + PLY parse +
     colour-map computation entirely off the
     main thread; results transferred back via
     Transferable ArrayBuffers.
     ════════════════════════════════════════════ */
  /* Worker code moved to static/js/ply-worker.js */

  /* ════════════════════════════════════════════
     Module state
     ════════════════════════════════════════════ */
  var _renderer, _scene, _cloud = null;
  var _lcam, _rcam, _lctrl, _rctrl;
  var _container = null;
  var _leftVP = null, _rightVP = null;
  var _currentBld = null, _pendingBld = null;
  var _syncEnabled = true, _syncBusy = false;
  var _leftMode = 'rgb', _rightMode = 'semantic';
  var _needsRender = false;
  var _visible = false;
  var _worker = null;

  function _mark() { _needsRender = true; }

  /* ════════════════════════════════════════════
     Init
     ════════════════════════════════════════════ */
  function init() {
    _container = document.querySelector('.dv-canvases');
    if (!_container) return;

    /* Single shared renderer — canvas is absolutely positioned to cover both panes */
    _renderer = new THREE.WebGLRenderer({ antialias: false });
    _renderer.setPixelRatio(1);
    var cv = _renderer.domElement;
    cv.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;';
    _container.appendChild(cv);

    /* Scene */
    _scene = new THREE.Scene();
    _scene.background = new THREE.Color(0xf4f5f8);

    /* Two independent cameras */
    _lcam = new THREE.PerspectiveCamera(52, 1, 0.01, 10000);
    _rcam = new THREE.PerspectiveCamera(52, 1, 0.01, 10000);
    [_lcam, _rcam].forEach(function(c) { c.position.set(0, -45, 22); c.up.set(0, 0, 1); });

    /* OrbitControls bound to the overlay divs (pointer-events pass through canvas) */
    var dvL = document.getElementById('dvLeft');
    var dvR = document.getElementById('dvRight');
    _lctrl = new THREE.OrbitControls(_lcam, dvL);
    _rctrl = new THREE.OrbitControls(_rcam, dvR);
    [_lctrl, _rctrl].forEach(function(c) {
      c.enableDamping = true; c.dampingFactor = 0.05;
      c.autoRotate = true;   c.autoRotateSpeed = 0.55;
      c.screenSpacePanning = false;
    });

    /* Camera sync */
    _lctrl.addEventListener('change', function() {
      if (_syncEnabled && !_syncBusy) {
        _syncBusy = true;
        _rcam.position.copy(_lcam.position);
        _rcam.quaternion.copy(_lcam.quaternion);
        _rctrl.target.copy(_lctrl.target);
        _syncBusy = false;
      }
      _mark();
    });
    _rctrl.addEventListener('change', function() {
      if (_syncEnabled && !_syncBusy) {
        _syncBusy = true;
        _lcam.position.copy(_rcam.position);
        _lcam.quaternion.copy(_rcam.quaternion);
        _lctrl.target.copy(_rctrl.target);
        _syncBusy = false;
      }
      _mark();
    });

    /* Resize observer keeps renderer buffer + viewport cache in sync */
    var ro = new ResizeObserver(_onResize);
    ro.observe(_container);
    _onResize();

    /* Stop rendering when viewer is scrolled off screen */
    var visObs = new IntersectionObserver(function(entries) {
      _visible = entries[0].isIntersecting;
      if (_visible) _mark();
    }, { threshold: 0 });
    visObs.observe(_container);

    /* PLY worker — dedicated file avoids blob: URL restrictions */
    try {
      _worker = new Worker('static/js/ply-worker.js?v=2');
      _worker.onerror = function(e) {
        console.error('HFX PLY worker error:', e);
        _showPlaceholder(_pendingBld || 'BLD009');
      };
    } catch(e) { console.warn('HFX viewer: Worker unavailable:', e); }

    _setupUI();
    _loop();
    loadBuilding('BLD009');
  }

  /* ════════════════════════════════════════════
     Render loop — render-on-demand.
     OrbitControls.update() returns true when
     the camera moves; we use that as the render
     trigger so idle frames cost nothing.
     ════════════════════════════════════════════ */
  var _lodActive = false, _lodTimer = null;
  function _loop() {
    requestAnimationFrame(_loop);
    if (!_visible) return;
    var lc = _lctrl ? _lctrl.update() : false;
    var rc = _rctrl ? _rctrl.update() : false;
    var camMoving = lc || rc;
    if (!camMoving && !_needsRender) return;
    /* Dynamic LOD: 25 % while camera moves, 100 % when still */
    if (_cloud) {
      var n = _cloud.geometry.attributes.position.count;
      if (camMoving) {
        _lodActive = true;
        clearTimeout(_lodTimer); _lodTimer = null;
      } else if (_lodActive && !_lodTimer) {
        _lodTimer = setTimeout(function() {
          _lodActive = false; _lodTimer = null; _needsRender = true;
        }, 200);
      }
      _cloud.geometry.setDrawRange(0, _lodActive ? Math.ceil(n * 0.25) : n);
    }
    _needsRender = false;
    _render();
  }

  function _render() {
    if (!_cloud || !_leftVP || !_rightVP) return;
    var lv = _leftVP, rv = _rightVP;
    if (lv.w < 1 || lv.h < 1 || rv.w < 1 || rv.h < 1) return;

    _renderer.setScissorTest(true);

    /* Left viewport — always RGB */
    _lcam.aspect = lv.w / lv.h;
    _lcam.updateProjectionMatrix();
    _renderer.setViewport(lv.x, lv.y, lv.w, lv.h);
    _renderer.setScissor (lv.x, lv.y, lv.w, lv.h);
    _cloud.material.uniforms.uMode.value = MODE_IDX[_leftMode] !== undefined ? MODE_IDX[_leftMode] : 0;
    _renderer.render(_scene, _lcam);

    /* Right viewport — user-selected mode */
    _rcam.aspect = rv.w / rv.h;
    _rcam.updateProjectionMatrix();
    _renderer.setViewport(rv.x, rv.y, rv.w, rv.h);
    _renderer.setScissor (rv.x, rv.y, rv.w, rv.h);
    _cloud.material.uniforms.uMode.value = MODE_IDX[_rightMode] !== undefined ? MODE_IDX[_rightMode] : 1;
    _renderer.render(_scene, _rcam);

    _renderer.setScissorTest(false);
  }

  function _onResize() {
    if (!_renderer || !_container) return;
    var W = _container.clientWidth, H = _container.clientHeight;
    if (!W || !H) return;
    /* updateStyle=false — we keep width/height:100% from init; only update GL buffer */
    _renderer.setSize(W, H, false);
    _updateViewports();
    _mark();
  }

  function _updateViewports() {
    var cv  = _renderer && _renderer.domElement;
    var dvL = document.getElementById('dvLeft');
    var dvR = document.getElementById('dvRight');
    if (!cv || !dvL || !dvR) return;
    var cR = cv.getBoundingClientRect();
    var lR = dvL.getBoundingClientRect();
    var rR = dvR.getBoundingClientRect();
    var cH = cR.height;
    /* Convert CSS coords (y-down) to WebGL coords (y-up from canvas bottom) */
    _leftVP  = { x: lR.left - cR.left, y: cH - (lR.bottom - cR.top), w: lR.width,  h: lR.height  };
    _rightVP = { x: rR.left - cR.left, y: cH - (rR.bottom - cR.top), w: rR.width, h: rR.height };
  }

  /* ════════════════════════════════════════════
     Building loader
     ════════════════════════════════════════════ */
  function loadBuilding(bldId) {
    if (bldId === _currentBld) return;
    _currentBld = bldId;
    _pendingBld  = bldId;

    var info  = BUILDINGS.filter(function(b) { return b.id === bldId; })[0];
    var title = document.getElementById('dvTitle');
    if (title && info)
      title.textContent = info.id + ' — ' + info.name + ' · ' + info.pts + ' pts · ' + info.cls + ' classes';

    document.querySelectorAll('.bsl-item').forEach(function(el) {
      el.classList.toggle('bsl-active', el.dataset.bld === bldId);
    });

    _updateLegend();
    _showLoading(true);

    if (!_worker) { _showPlaceholder(bldId); return; }

    var id  = bldId + '_' + Date.now();
    var url = 'https://hfx3d-ply.zazukhan58.workers.dev/' + bldId + '_preview.ply';

    function handler(e) {
      if (e.data.id !== id) return;
      _worker.removeEventListener('message', handler);
      if (_pendingBld !== bldId) return; /* superseded by a later click */
      e.data.error ? _showPlaceholder(bldId) : _ingest(e.data);
    }
    _worker.addEventListener('message', handler);
    _worker.postMessage({ url: url, id: id });
  }

  /* Called on the main thread once the Worker transfers all data */
  function _ingest(d) {
    /* Dispose previous cloud */
    if (_cloud) {
      _scene.remove(_cloud);
      _cloud.geometry.dispose();
      _cloud.material.dispose();
      _cloud = null;
    }

    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(d.xyz,  3));
    geo.setAttribute('aRgb',     new THREE.BufferAttribute(d.rgb,  3));
    geo.setAttribute('aSem',     new THREE.BufferAttribute(d.sem,  3));
    geo.setAttribute('aInt',     new THREE.BufferAttribute(d.intC, 3));
    geo.setAttribute('aTim',     new THREE.BufferAttribute(d.timC, 3));

    geo.computeBoundingBox();
    var cen = new THREE.Vector3();
    geo.boundingBox.getCenter(cen);
    geo.translate(-cen.x, -cen.y, -cen.z);

    _cloud = new THREE.Points(geo, new THREE.ShaderMaterial({
      vertexShader:   VERT_SRC,
      fragmentShader: FRAG_SRC,
      uniforms: { uMode: { value: 0 } },
    }));
    _scene.add(_cloud);

    var box  = new THREE.Box3().setFromObject(_cloud);
    var size = box.getSize(new THREE.Vector3()).length();
    [_lcam, _rcam].forEach(function(cam) {
      cam.position.set(0, -size * 0.72, size * 0.36);
      cam.far = size * 25;
      cam.updateProjectionMatrix();
    });
    [_lctrl, _rctrl].forEach(function(ctrl) {
      ctrl.target.set(0, 0, 0);
      ctrl.update();
      ctrl.saveState();
    });
    _rcam.position.copy(_lcam.position);
    _rcam.quaternion.copy(_lcam.quaternion);
    _rctrl.target.copy(_lctrl.target);

    document.querySelectorAll('.dv-mode-btn').forEach(function(btn) {
      var m = btn.dataset.mode;
      btn.classList.toggle('dv-mode-btn-dim',
        (m === 'intensity' && !d.hI) || (m === 'time' && !d.hG));
    });

    if ((_rightMode === 'intensity' && !d.hI) || (_rightMode === 'time' && !d.hG))
      _rightMode = 'semantic';

    _updateLegend();
    _showLoading(false);
    _mark();
  }

  /* ════════════════════════════════════════════
     UI wiring
     ════════════════════════════════════════════ */
  function _setupUI() {
    var btnAR    = document.getElementById('dvAutoRotate');
    var btnReset = document.getElementById('dvReset');
    var btnSync  = document.getElementById('dvSync');

    if (btnAR) btnAR.addEventListener('click', function() {
      var on = !_lctrl.autoRotate;
      _lctrl.autoRotate = _rctrl.autoRotate = on;
      btnAR.classList.toggle('active', on);
    });

    if (btnReset) btnReset.addEventListener('click', function() {
      _lctrl.reset(); _rctrl.reset(); _mark();
    });

    if (btnSync) btnSync.addEventListener('click', function() {
      _syncEnabled = !_syncEnabled;
      btnSync.classList.toggle('active', _syncEnabled);
    });

    document.querySelectorAll('.dv-zoom-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var f = btn.dataset.dir === '+' ? 0.82 : 1.22;
        _lcam.position.multiplyScalar(f); _lctrl.update();
        _rcam.position.multiplyScalar(f); _rctrl.update();
        _mark();
      });
    });

    var lblEl = document.getElementById('dvRightLabel');
    document.querySelectorAll('.dv-mode-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        document.querySelectorAll('.dv-mode-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        _rightMode = btn.dataset.mode;
        if (lblEl) {
          var icon = btn.querySelector('i');
          var text = (btn.querySelector('.dv-mode-text') || btn).textContent.trim();
          lblEl.innerHTML = (icon ? '<i class="' + icon.className + '"></i> ' : '') + text;
        }
        _mark();
      });
    });

    var track = document.getElementById('bslTrack');
    if (track) {
      var origItems = Array.prototype.slice.call(track.querySelectorAll('.bsl-item'));
      origItems.forEach(function(item) {
        var clone = item.cloneNode(true);
        clone.setAttribute('aria-hidden', 'true');
        track.appendChild(clone);
      });
      track.addEventListener('click', function(e) {
        var item = e.target.closest('.bsl-item');
        if (item && item.dataset.bld) loadBuilding(item.dataset.bld);
      });
    }
  }

  /* ── Overlay helpers ── */
  function _showLoading(on) {
    var el = document.getElementById('dvLoading');
    var ph = document.getElementById('dvPlaceholder');
    if (el) el.style.display = on ? 'flex' : 'none';
    if (ph && on) ph.style.display = 'none';
  }

  function _showPlaceholder(bldId) {
    _showLoading(false);
    var ph   = document.getElementById('dvPlaceholder');
    var code = document.getElementById('dvPlaceholderCmd');
    if (ph)   ph.style.display = 'flex';
    if (code) code.textContent = 'python tools/export_ply.py data/hfx3d/HFX_' + bldId + '_ZEB_CLEAN.las';
  }

  function _updateLegend() {
    var el = document.getElementById('dvLegend');
    if (!el) return;
    el.innerHTML = '';
    CLS_NAMES.forEach(function(name, l) {
      var item = document.createElement('div');
      item.className = 'dv-legend-item';
      item.innerHTML = '<span class="dv-lc" style="background:' + CLS_HEX[l] + '"></span>' + name;
      el.appendChild(item);
    });
  }

  /* ── Public API ── */
  window.loadBuildingViewer = function(bldId) {
    loadBuilding(bldId);
    var sec = document.getElementById('explorer');
    if (sec) sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  /* ── Boot ── */
  function _safeInit() {
    try { init(); }
    catch(e) {
      console.error('HFX viewer init failed:', e);
      var ph = document.getElementById('dvPlaceholder');
      var code = document.getElementById('dvPlaceholderCmd');
      if (ph) { ph.style.display = 'flex'; }
      if (code) { code.textContent = 'Error: ' + e.message; }
    }
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', _safeInit);
  else
    _safeInit();

})();
