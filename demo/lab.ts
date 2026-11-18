// <canyon-lab>: flood a photogrammetry model of the Kojori canyon between two
// movable dams and read the water volume. The terrain is a height grid (DSM)
// in metres, georeferenced with the flight controller's GPS; the volume is a
// flood fill over that same grid, so the number is computed from exactly what
// is drawn. Progressive enhancement like <ne-lab>: the light DOM carries the
// static fallback figure, hidden once this module defines the element.
// Three.js is the one library here and is imported only when the lab scrolls
// into view (see DESIGN.md, "Interactive labs").
import css from './lab.css'

type Lang = 'en' | 'ru' | 'ja'
const BASE = new URL('./data/', import.meta.url).href

const STRINGS = {
  en: {
    wallA: 'upper dam', wallB: 'lower dam', level: 'water level', track: 'flight track', holes: 'show filled holes',
    volume: 'volume', area: 'surface', depth: 'max depth', mean: 'mean depth', dam: 'dam', crest: 'crest', high: 'high',
    msl: 'm a.s.l.', leak: 'the water spills out of the model: lower the level or move a dam',
    dry: 'the level is below the canyon floor', interp: 'over interpolated terrain', curve: 'volume vs level',
    loading: 'loading the terrain…', failed: 'the 3D view could not start (WebGL is unavailable)',
    hint: 'drag to orbit · wheel to zoom · right-drag to pan', mln: 'million m³', aria: 'Interactive 3D model of the Kojori canyon with two dams and an adjustable water level',
  },
  ru: {
    wallA: 'верхняя плотина', wallB: 'нижняя плотина', level: 'уровень воды', track: 'трек полёта', holes: 'показать заполненные дыры',
    volume: 'объём', area: 'зеркало', depth: 'макс. глубина', mean: 'средняя глубина', dam: 'плотина', crest: 'гребень', high: 'высота',
    msl: 'м над у. м.', leak: 'вода выливается за пределы модели: опустите уровень или сдвиньте плотину',
    dry: 'уровень ниже дна каньона', interp: 'над интерполированным рельефом', curve: 'объём от уровня',
    loading: 'загружаю рельеф…', failed: 'не удалось запустить 3D (WebGL недоступен)',
    hint: 'тяните, чтобы вращать · колесо: масштаб · правая кнопка: сдвиг', mln: 'млн м³', aria: 'Интерактивная 3D-модель каньона Коджори с двумя плотинами и регулируемым уровнем воды',
  },
  ja: {
    wallA: '上流のダム', wallB: '下流のダム', level: '水位', track: '飛行経路', holes: '補間した穴を表示',
    volume: '体積', area: '水面', depth: '最大水深', mean: '平均水深', dam: 'ダム', crest: '堤頂長', high: '高さ',
    msl: 'm（海抜）', leak: '水がモデルの外へあふれています。水位を下げるかダムを動かしてください',
    dry: '水位が谷底より低い', interp: '補間した地形の上', curve: '水位と体積',
    loading: '地形を読み込み中…', failed: '3D表示を開始できません（WebGLが使えません）',
    hint: 'ドラッグで回転 · ホイールで拡大 · 右ドラッグで移動', mln: '百万 m³', aria: '二つのダムと水位を調整できるコジョリ渓谷のインタラクティブ3Dモデル',
  },
} satisfies Record<Lang, Record<string, string>>

interface Meta {
  w: number; h: number; cell: number; e0: number; n1: number; zmin: number; zstep: number
  home: { lat: number; lon: number; alt: number }
  axis: [number, number][]; cams: [number, number, number][]; track?: [number, number, number][]
  georef: { rms: number; scale: number; lag: number; cams: number }; measured: number
}
interface Fill { volume: number; area: number; maxDepth: number; interp: number; leak: boolean; n: number }
interface Dam { crest: number; height: number }

const AXIS_STEP = 10 // metres between axis samples in meta.json
const MIN_GAP = 40 // metres between the dams
const fmt = (v: number, d = 0) => v.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }).replace(/,/g, ' ')

class CanyonLab extends HTMLElement {
  private t = STRINGS.en as Record<string, string>
  private meta!: Meta
  private Z!: Float32Array // height above home, row 0 = north edge
  private valid!: Uint8Array
  private mark!: Uint8Array // flood-fill result at grid resolution
  private queue!: Int32Array
  private sA = 620
  private sB = 1500
  private level = -80
  private els: Record<string, HTMLElement> = {}
  private three: any
  private dirty = true
  private started = false
  private curveJob = 0

  connectedCallback() {
    if (this.started) return
    this.started = true
    const lang = (document.documentElement.lang || 'en').slice(0, 2) as Lang
    this.t = STRINGS[lang] ?? STRINGS.en
    if (!document.getElementById('canyon-lab-css')) {
      const st = document.createElement('style')
      st.id = 'canyon-lab-css'
      st.textContent = css
      document.head.appendChild(st)
    }
    this.build()
    const io = new IntersectionObserver((e) => {
      if (e.some((x) => x.isIntersecting)) {
        io.disconnect()
        this.start().catch((err) => {
          console.error('[canyon-lab]', err)
          this.els.status.textContent = this.t.failed
          this.classList.add('failed') // the fallback figure comes back (lab.css)
        })
      }
    }, { rootMargin: '300px' })
    io.observe(this)
  }

  // ---------------------------------------------------------------- DOM
  private build() {
    const t = this.t
    const root = document.createElement('div')
    root.className = 'lab'
    root.innerHTML = `
      <div class="view" role="img" aria-label="${t.aria}"><div class="status">${t.loading}</div><div class="warn" hidden></div><div class="hint">${t.hint}</div></div>
      <div class="ctl">
        <label><span>${t.wallA} <output data-o="sA"></output></span><input type="range" data-i="sA" step="10"></label>
        <label><span>${t.wallB} <output data-o="sB"></output></span><input type="range" data-i="sB" step="10"></label>
        <label class="lvl"><span>${t.level} <output data-o="level"></output></span><input type="range" data-i="level" step="0.5"></label>
      </div>
      <div class="out">
        <div class="big"><span class="k">${t.volume}</span><span class="v" data-o="vol"></span><span class="u" data-o="volu"></span></div>
        <dl>
          <div><dt>${t.area}</dt><dd data-o="area"></dd></div>
          <div><dt>${t.depth}</dt><dd data-o="depth"></dd></div>
          <div><dt>${t.mean}</dt><dd data-o="mean"></dd></div>
          <div><dt>${t.dam} A</dt><dd data-o="damA"></dd></div>
          <div><dt>${t.dam} B</dt><dd data-o="damB"></dd></div>
          <div><dt>${t.interp}</dt><dd data-o="interp"></dd></div>
        </dl>
        <figure class="curve"><svg viewBox="0 0 240 90" preserveAspectRatio="none" aria-hidden="true"><path class="area"/><path class="line"/><line class="now"/><circle class="dot" r="2.6"/></svg><figcaption>${t.curve}</figcaption></figure>
      </div>
      <div class="opt">
        <label><input type="checkbox" data-i="track" checked> ${t.track}</label>
        <label><input type="checkbox" data-i="holes"> ${t.holes}</label>
        <span class="geo" data-o="geo"></span>
      </div>`
    this.appendChild(root)
    root.querySelectorAll<HTMLElement>('[data-o],[data-i]').forEach((e) => (this.els[e.dataset.o ?? 'i_' + e.dataset.i] = e))
    this.els.view = root.querySelector('.view')!
    this.els.status = root.querySelector('.status')!
    this.els.warn = root.querySelector('.warn')!
    this.els.svg = root.querySelector('svg')! as unknown as HTMLElement
  }

  // ---------------------------------------------------------------- data
  private async start() {
    const [meta, dsm, THREE, oc] = await Promise.all([
      fetch(BASE + 'meta.json').then((r) => r.json() as Promise<Meta>),
      loadImage(BASE + 'dsm.png'),
      import('three'),
      import('three/addons/controls/OrbitControls.js'),
    ])
    this.meta = meta
    const { w, h } = meta
    const cv = document.createElement('canvas')
    cv.width = w
    cv.height = h
    const cx = cv.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' })!
    cx.drawImage(dsm, 0, 0)
    const px = cx.getImageData(0, 0, w, h).data
    this.Z = new Float32Array(w * h)
    this.valid = new Uint8Array(w * h)
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      this.Z[i] = meta.zmin + ((px[j] << 8) | px[j + 1]) * meta.zstep // R = high byte, G = low byte
      this.valid[i] = px[j + 2] > 200 ? 2 : px[j + 2] > 60 ? 1 : 0 // 2 measured, 1 filled hole, 0 outside the model
    }
    this.mark = new Uint8Array(w * h)
    this.queue = new Int32Array(w * h)

    const L = (meta.axis.length - 1) * AXIS_STEP
    this.sA = clamp(this.sA, 0, L - MIN_GAP)
    this.sB = clamp(this.sB, this.sA + MIN_GAP, L)
    const floorB = this.floorAt(this.sB)
    this.level = Math.round(floorB + 55)
    const iA = this.els.i_sA as HTMLInputElement, iB = this.els.i_sB as HTMLInputElement, iL = this.els.i_level as HTMLInputElement
    for (const i of [iA, iB]) { i.min = '0'; i.max = String(L) }
    iL.min = String(Math.floor(this.floorAt(L) - 5))
    iL.max = String(Math.ceil(this.floorAt(0) + 140))
    iA.value = String(this.sA); iB.value = String(this.sB); iL.value = String(this.level)
    iA.addEventListener('input', () => { this.sA = Math.min(+iA.value, this.sB - MIN_GAP); iA.value = String(this.sA); this.update(true) })
    iB.addEventListener('input', () => { this.sB = Math.max(+iB.value, this.sA + MIN_GAP); iB.value = String(this.sB); this.update(true) })
    iL.addEventListener('input', () => { this.level = +iL.value; this.update(false) })
    this.els.geo.textContent = `GPS fit: ${meta.georef.cams} cams · rms ${meta.georef.rms} m · grid ${meta.cell} m`

    this.initScene(THREE, oc.OrbitControls, await loadImage(BASE + 'ortho.jpg'))
    this.els.status.remove()
    this.update(true)
  }

  private idx(e: number, n: number) {
    const { w, h, cell, e0, n1 } = this.meta
    const c = Math.floor((e - e0) / cell), r = Math.floor((n1 - n) / cell)
    return c < 0 || r < 0 || c >= w || r >= h ? -1 : r * w + c
  }

  /** Bilinear terrain height at ENU metres (NaN outside the grid). */
  private zAt(e: number, n: number) {
    const { w, h, cell, e0, n1 } = this.meta
    const x = (e - e0) / cell - 0.5, y = (n1 - n) / cell - 0.5
    if (x < 0 || y < 0 || x > w - 1.001 || y > h - 1.001) return NaN
    const c = Math.floor(x), r = Math.floor(y), fx = x - c, fy = y - r, i = r * w + c, Z = this.Z
    return (Z[i] * (1 - fx) + Z[i + 1] * fx) * (1 - fy) + (Z[i + w] * (1 - fx) + Z[i + w + 1] * fx) * fy
  }

  /** Axis point and unit tangent at station s (metres along the canyon). */
  private axisAt(s: number) {
    const a = this.meta.axis, f = clamp(s / AXIS_STEP, 0, a.length - 1.001), i = Math.floor(f), u = f - i
    const p = [a[i][0] + (a[i + 1][0] - a[i][0]) * u, a[i][1] + (a[i + 1][1] - a[i][1]) * u]
    const j0 = Math.max(0, i - 3), j1 = Math.min(a.length - 1, i + 4) // tangent over ~70 m: the wall must not twitch
    const tx = a[j1][0] - a[j0][0], ty = a[j1][1] - a[j0][1], l = Math.hypot(tx, ty)
    return { e: p[0], n: p[1], te: tx / l, tn: ty / l }
  }

  private floorAt(s: number) {
    const p = this.axisAt(s)
    let z = Infinity // the axis is smoothed, so look for the floor a little to both sides
    for (let d = -40; d <= 40; d += 4) { const v = this.zAt(p.e - p.tn * d, p.n + p.te * d); if (v < z) z = v }
    return z
  }

  // ---------------------------------------------------------------- flood fill
  /** Water below `level`, connected to the canyon floor, between the two infinite vertical walls. */
  private flood(level: number, keep: boolean): Fill {
    const { w, h, cell } = this.meta, Z = this.Z, mark = this.mark, q = this.queue
    const A = this.axisAt(this.sA), B = this.axisAt(this.sB)
    mark.fill(0)
    let head = 0, tail = 0
    const inside = (i: number) => {
      const e = this.meta.e0 + ((i % w) + 0.5) * cell, n = this.meta.n1 - (Math.floor(i / w) + 0.5) * cell
      return (e - A.e) * A.te + (n - A.n) * A.tn >= 0 && (e - B.e) * B.te + (n - B.n) * B.tn <= 0
    }
    for (let s = this.sA + 4; s < this.sB; s += 4) { // seeds: the floor line between the dams
      const p = this.axisAt(s)
      for (let d = -40; d <= 40; d += 4) {
        const i = this.idx(p.e - p.tn * d, p.n + p.te * d)
        if (i >= 0 && !mark[i] && Z[i] < level && inside(i)) { mark[i] = 1; q[tail++] = i }
      }
    }
    let vol = 0, interp = 0, maxD = 0, leak = false
    while (head < tail) {
      const i = q[head++], c = i % w, r = (i - c) / w, d = level - Z[i]
      vol += d
      if (d > maxD) maxD = d
      if (this.valid[i] < 2) interp += d
      if (!this.valid[i] || c === 0 || r === 0 || c === w - 1 || r === h - 1) { leak = true; continue }
      for (const j of [i - 1, i + 1, i - w, i + w]) {
        if (!mark[j] && Z[j] < level && inside(j)) { mark[j] = 1; q[tail++] = j }
      }
    }
    const a2 = cell * cell
    if (!keep) mark.fill(0)
    return { volume: vol * a2, area: tail * a2, maxDepth: maxD, interp: vol > 0 ? interp / vol : 0, leak, n: tail }
  }

  /** Cross-section under a wall: terrain profile from the floor out to where it rises above the water. */
  private section(s: number, level: number) {
    const p = this.axisAt(s), ne = -p.tn, nn = p.te, step = this.meta.cell
    let d0 = 0, zf = Infinity
    for (let d = -40; d <= 40; d += 2) { const v = this.zAt(p.e + ne * d, p.n + nn * d); if (v < zf) { zf = v; d0 = d } }
    const side = (dir: number) => {
      const pts: [number, number, number, number][] = []
      for (let d = d0; Math.abs(d - d0) < 1600; d += dir * step) {
        const e = p.e + ne * d, n = p.n + nn * d, z = this.zAt(e, n)
        if (Number.isNaN(z)) break
        pts.push([e, n, z, d])
        if (z >= level) break
      }
      return pts
    }
    const pts = [...side(-1).reverse(), ...side(1).slice(1)]
    return { pts, floor: zf, wet: zf < level, p, ne, nn, d0 }
  }

  // ---------------------------------------------------------------- scene
  private initScene(THREE: any, OrbitControls: any, ortho: HTMLImageElement) {
    const { w, h, cell, e0, n1 } = this.meta, view = this.els.view
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    view.prepend(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1.6, 5, 20000)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.maxPolarAngle = Math.PI * 0.49
    controls.minDistance = 120
    controls.maxDistance = 5000
    controls.addEventListener('change', () => (this.dirty = true))

    // Terrain: the DSM decimated to <= ~330k vertices. World = (east, up, -north) in metres from home.
    const k = Math.max(2, Math.ceil(Math.sqrt((w * h) / (matchMedia('(max-width: 720px)').matches ? 120000 : 330000))))
    const gw = Math.floor((w - 1) / k) + 1, gh = Math.floor((h - 1) / k) + 1
    const pos = new Float32Array(gw * gh * 3), uv = new Float32Array(gw * gh * 2), tz = new Float32Array(gw * gh), ok = new Float32Array(gw * gh)
    const cells = new Int32Array(gw * gh)
    for (let r = 0, v = 0; r < gh; r++) for (let c = 0; c < gw; c++, v++) {
      const i = r * k * w + c * k
      cells[v] = i
      pos[v * 3] = e0 + (c * k + 0.5) * cell; pos[v * 3 + 1] = this.Z[i]; pos[v * 3 + 2] = -(n1 - (r * k + 0.5) * cell)
      uv[v * 2] = (c * k + 0.5) / w; uv[v * 2 + 1] = 1 - (r * k + 0.5) / h
      tz[v] = this.Z[i]; ok[v] = this.valid[i] / 2
    }
    const index = new Uint32Array((gw - 1) * (gh - 1) * 6)
    for (let r = 0, n = 0; r < gh - 1; r++) for (let c = 0; c < gw - 1; c++) {
      const a = r * gw + c, b = a + 1, d = a + gw, e = d + 1
      index[n++] = a; index[n++] = d; index[n++] = b; index[n++] = b; index[n++] = d; index[n++] = e
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
    geo.setAttribute('ok', new THREE.BufferAttribute(ok, 1))
    geo.setIndex(new THREE.BufferAttribute(index, 1))
    geo.computeVertexNormals()
    const tex = new THREE.Texture(ortho)
    tex.colorSpace = THREE.SRGBColorSpace
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy()
    tex.needsUpdate = true
    const holes = { value: 0 }
    const land = new THREE.MeshLambertMaterial({ map: tex })
    land.onBeforeCompile = (sh: any) => { // optional overlay: cells with no measurement (inpainted) get a hatch
      sh.uniforms.uHoles = holes
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float ok; varying float vOk; varying vec3 vW;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvOk = ok; vW = position;')
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uHoles; varying float vOk; varying vec3 vW;')
        .replace('#include <dithering_fragment>', `#include <dithering_fragment>
          if (vOk < 0.2) discard;
          float hatch = step(0.5, fract((vW.x + vW.z) / 14.0));
          gl_FragColor.rgb = mix(gl_FragColor.rgb, mix(gl_FragColor.rgb, vec3(0.82, 0.25, 0.55), 0.35 + 0.3 * hatch), uHoles * (1.0 - smoothstep(0.7, 0.9, vOk)));`)
    }
    scene.add(new THREE.Mesh(geo, land))
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 2.1))
    const sun = new THREE.DirectionalLight(0xfff1dc, 1.5)
    sun.position.set(-0.6, 0.55, 0.5) // the photo already carries the low afternoon sun; this only adds relief
    scene.add(sun)

    // Water: the same grid at the water level; the fragment shader keeps only flooded cells and shades by depth.
    const wgeo = new THREE.BufferGeometry()
    wgeo.setAttribute('position', new THREE.BufferAttribute(pos.slice(), 3))
    wgeo.setAttribute('tz', new THREE.BufferAttribute(tz, 1))
    const wet = new THREE.BufferAttribute(new Float32Array(gw * gh), 1)
    wet.setUsage(THREE.DynamicDrawUsage)
    wgeo.setAttribute('wet', wet)
    wgeo.setIndex(new THREE.BufferAttribute(index, 1))
    const uni = { uLevel: { value: 0 }, uTime: { value: 0 }, uSun: { value: new THREE.Vector3(-0.6, 0.55, 0.5).normalize() } }
    const water = new THREE.Mesh(wgeo, new THREE.ShaderMaterial({
      uniforms: uni, transparent: true, depthWrite: false, side: THREE.DoubleSide,
      vertexShader: `
        attribute float tz; attribute float wet; uniform float uLevel;
        varying float vDepth; varying float vWet; varying vec3 vP;
        void main() { vDepth = uLevel - tz; vWet = wet; vec3 p = vec3(position.x, uLevel, position.z); vP = p;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }`,
      fragmentShader: `
        uniform float uTime; uniform vec3 uSun; varying float vDepth; varying float vWet; varying vec3 vP;
        float wave(vec2 p, float t) { return sin(p.x * 0.11 + t) * 0.5 + sin(p.y * 0.17 - t * 1.3) * 0.3 + sin((p.x + p.y) * 0.31 + t * 2.1) * 0.2; }
        void main() {
          if (vWet < 0.5 || vDepth <= 0.0) discard;
          vec2 q = vP.xz; float e = 0.6, t = uTime;
          vec3 n = normalize(vec3(wave(q - vec2(e, 0.0), t) - wave(q + vec2(e, 0.0), t), 5.0, wave(q - vec2(0.0, e), t) - wave(q + vec2(0.0, e), t)));
          vec3 v = normalize(cameraPosition - vP);
          float fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
          float spec = pow(max(dot(reflect(-uSun, n), v), 0.0), 90.0);
          float d = 1.0 - exp(-vDepth / 22.0);                                   // Beer-Lambert-ish: deeper = darker, more opaque
          vec3 col = mix(vec3(0.20, 0.66, 0.70), vec3(0.02, 0.16, 0.33), d);
          col = mix(col, vec3(0.74, 0.86, 0.95), fres * 0.6) + spec * 0.8;
          float shore = smoothstep(1.2, 0.0, vDepth) * (0.6 + 0.4 * sin(t * 2.0 + (q.x + q.y) * 0.4));
          col = mix(col, vec3(0.95), shore * 0.55);
          gl_FragColor = vec4(col, mix(0.42, 0.9, d) + fres * 0.1);
        }`,
    }))
    water.renderOrder = 2
    scene.add(water)

    // Dams: a translucent slab under each wall plus a line draped over the terrain along the wall.
    const mkDam = (color: number) => {
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false }))
      const crest = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff }))
      const drape = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color }))
      mesh.renderOrder = 3; crest.renderOrder = 4; drape.renderOrder = 4
      scene.add(mesh, crest, drape)
      return { mesh, crest, drape }
    }
    const dams = [mkDam(0x8f86ff), mkDam(0xf0a45a)]

    // Flight track (GPS) and the SfM camera centres that tied the model to it.
    const trackGroup = new THREE.Group()
    if (this.meta.track) {
      const g = new THREE.BufferGeometry().setFromPoints(this.meta.track.map((p) => new THREE.Vector3(p[0], p[2], -p[1])))
      trackGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.75 })))
    }
    const cg = new THREE.BufferGeometry().setFromPoints(this.meta.cams.map((p) => new THREE.Vector3(p[0], p[2], -p[1])))
    trackGroup.add(new THREE.Points(cg, new THREE.PointsMaterial({ color: 0xff5d5d, size: 2.5, sizeAttenuation: false })))
    scene.add(trackGroup)

    const mid = this.axisAt((this.sA + this.sB) / 2)
    controls.target.set(mid.e, this.floorAt((this.sA + this.sB) / 2) + 30, -mid.n)
    camera.position.set(mid.e + 900, 430, -mid.n + 760)
    controls.update()

    const resize = () => {
      const wv = view.clientWidth, hv = view.clientHeight
      renderer.setSize(wv, hv, false)
      camera.aspect = wv / hv
      camera.updateProjectionMatrix()
      this.dirty = true
    }
    new ResizeObserver(resize).observe(view)
    resize()

    const calm = matchMedia('(prefers-reduced-motion: reduce)').matches
    let visible = true
    new IntersectionObserver((e) => (visible = e[0].isIntersecting)).observe(view)
    const t0 = performance.now()
    renderer.setAnimationLoop(() => {
      if (!visible) return
      if (!calm) uni.uTime.value = (performance.now() - t0) / 1000
      if (controls.update() || this.dirty || !calm) { renderer.render(scene, camera); this.dirty = false }
    })

    ;(this.els.i_track as HTMLInputElement).addEventListener('change', (e) => { trackGroup.visible = (e.target as HTMLInputElement).checked; this.dirty = true })
    ;(this.els.i_holes as HTMLInputElement).addEventListener('change', (e) => { holes.value = (e.target as HTMLInputElement).checked ? 1 : 0; this.dirty = true })
    this.three = { THREE, uni, wet, cells, dams, renderer, scene, camera, controls }
  }

  private drawDam(d: any, s: number, level: number): Dam {
    const { THREE } = this.three, sec = this.section(s, level), up = 1.2
    const V = (e: number, n: number, z: number) => new THREE.Vector3(e, z, -n)
    const span = sec.wet ? sec.pts : [-160, -80, -40, -20, 0, 20, 40, 80, 160].map((x) => {
      const e = sec.p.e + sec.ne * (sec.d0 + x), n = sec.p.n + sec.nn * (sec.d0 + x)
      return [e, n, this.zAt(e, n), 0] as [number, number, number, number]
    }).filter((p) => !Number.isNaN(p[2]))
    d.drape.geometry.dispose()
    d.drape.geometry = new THREE.BufferGeometry().setFromPoints(span.map((p: number[]) => V(p[0], p[1], p[2] + up)))
    d.mesh.visible = d.crest.visible = sec.wet && sec.pts.length > 1
    if (!d.mesh.visible) return { crest: 0, height: 0 }
    const top = level + 2, n = sec.pts.length, pos = new Float32Array(n * 6), idx: number[] = []
    sec.pts.forEach((p, i) => {
      pos.set([p[0], Math.min(p[2], top), -p[1], p[0], top, -p[1]], i * 6)
      if (i < n - 1) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2)
    })
    d.mesh.geometry.dispose()
    d.mesh.geometry = new THREE.BufferGeometry()
    d.mesh.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    d.mesh.geometry.setIndex(idx)
    d.crest.geometry.dispose()
    d.crest.geometry = new THREE.BufferGeometry().setFromPoints([V(sec.pts[0][0], sec.pts[0][1], top), V(sec.pts[n - 1][0], sec.pts[n - 1][1], top)])
    return { crest: Math.abs(sec.pts[n - 1][3] - sec.pts[0][3]), height: level - sec.floor }
  }

  // ---------------------------------------------------------------- update
  private update(wallsMoved: boolean) {
    if (!this.three) return
    const { uni, wet, cells, dams } = this.three, t = this.t, o = this.els, alt = this.meta.home.alt
    const f = this.flood(this.level, true)
    const arr = wet.array as Float32Array
    for (let v = 0; v < cells.length; v++) arr[v] = this.mark[cells[v]]
    wet.needsUpdate = true
    uni.uLevel.value = this.level
    const dA = this.drawDam(dams[0], this.sA, this.level), dB = this.drawDam(dams[1], this.sB, this.level)

    o.sA.textContent = `${fmt(this.sA)} m`
    o.sB.textContent = `${fmt(this.sB)} m`
    o.level.textContent = `${fmt(alt + this.level, 1)} ${t.msl}`
    const big = f.volume >= 1e6
    o.vol.textContent = big ? fmt(f.volume / 1e6, 2) : fmt(f.volume)
    o.volu.textContent = big ? t.mln : 'm³'
    o.area.textContent = f.area >= 1e4 ? `${fmt(f.area / 1e4, 2)} ha` : `${fmt(f.area)} m²`
    o.depth.textContent = `${fmt(f.maxDepth, 1)} m`
    o.mean.textContent = `${fmt(f.area ? f.volume / f.area : 0, 1)} m`
    const dam = (d: Dam) => (d.crest ? `${t.crest} ${fmt(d.crest)} m · ${t.high} ${fmt(d.height, 1)} m` : '—')
    o.damA.textContent = dam(dA)
    o.damB.textContent = dam(dB)
    o.interp.textContent = `${fmt(f.interp * 100, 1)} %`
    const warn = f.leak ? t.leak : f.n === 0 ? t.dry : ''
    o.warn.textContent = warn
    o.warn.hidden = !warn
    this.classList.toggle('leak', f.leak)
    this.dirty = true
    if (wallsMoved) this.scheduleCurve()
    else this.placeCurveMarker(f.volume)
  }

  // Volume-versus-level curve for the current dams: 40 flood fills, done off the input path.
  private curve: { lv: number[]; vol: number[]; leakFrom: number } | null = null
  private scheduleCurve() {
    cancelAnimationFrame(this.curveJob)
    this.curveJob = requestAnimationFrame(() => setTimeout(() => {
      const iL = this.els.i_level as HTMLInputElement, lo = +iL.min, hi = +iL.max, lv: number[] = [], vol: number[] = []
      let leakFrom = Infinity
      for (let i = 0; i <= 40; i++) {
        const l = lo + ((hi - lo) * i) / 40, f = this.flood(l, false)
        if (f.leak) { leakFrom = l; break }
        lv.push(l); vol.push(f.volume)
      }
      this.curve = { lv, vol, leakFrom }
      this.flood(this.level, true) // restore the mask of the current level
      const svg = this.els.svg, vmax = Math.max(...vol, 1), X = (l: number) => ((l - lo) / (hi - lo)) * 240, Y = (v: number) => 86 - (v / vmax) * 80
      const d = lv.map((l, i) => `${i ? 'L' : 'M'}${X(l).toFixed(1)},${Y(vol[i]).toFixed(1)}`).join('')
      svg.querySelector('.line')!.setAttribute('d', d)
      svg.querySelector('.area')!.setAttribute('d', lv.length ? `${d}L${X(lv[lv.length - 1]).toFixed(1)},86L${X(lv[0]).toFixed(1)},86Z` : '')
      this.placeCurveMarker(this.flood(this.level, true).volume)
    }, 0))
  }

  private placeCurveMarker(volume: number) {
    const c = this.curve, svg = this.els.svg
    if (!c || !c.lv.length) return
    const iL = this.els.i_level as HTMLInputElement, lo = +iL.min, hi = +iL.max, vmax = Math.max(...c.vol, 1)
    const x = ((this.level - lo) / (hi - lo)) * 240, y = 86 - (Math.min(volume, vmax) / vmax) * 80
    const now = svg.querySelector('.now')!, dot = svg.querySelector('.dot')!
    now.setAttribute('x1', String(x)); now.setAttribute('x2', String(x)); now.setAttribute('y1', '4'); now.setAttribute('y2', '86')
    dot.setAttribute('cx', String(x)); dot.setAttribute('cy', String(y))
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((res, rej) => {
    const im = new Image()
    im.onload = () => res(im)
    im.onerror = () => rej(new Error('failed to load ' + src))
    im.src = src
  })
}

if (!customElements.get('canyon-lab')) customElements.define('canyon-lab', CanyonLab)
