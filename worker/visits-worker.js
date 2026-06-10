/**
 * hfx3d-visits — visitor tracking worker for the HFX3D project page.
 *
 * POST /visit     record a visitor (geo from request.cf, dedup server-side)
 * GET  /visitors  { total, visitors: [{lat, lon, city, country}] }, cached 5 min
 *
 * Secrets (wrangler secret put): SUPABASE_URL, SUPABASE_SERVICE_KEY, HASH_SALT
 */

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(data, extra, status) {
  return new Response(JSON.stringify(data), {
    status: status || 200,
    headers: Object.assign({ 'Content-Type': 'application/json' }, CORS, extra || {}),
  });
}

async function sha256Hex(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function geoFrom(request) {
  const cf = request.cf || {};
  const lat = parseFloat(cf.latitude);
  const lon = parseFloat(cf.longitude);
  if (!isFinite(lat) || !isFinite(lon)) return null;
  if (lat === 0 && lon === 0) return null; // null island
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return { lat, lon, city: cf.city || null, country: cf.country || null };
}

function sbHeaders(env, extra) {
  return Object.assign(
    {
      apikey: env.SUPABASE_SERVICE_KEY,
      Authorization: 'Bearer ' + env.SUPABASE_SERVICE_KEY,
      'Content-Type': 'application/json',
    },
    extra || {}
  );
}

async function handleVisit(request, env, ctx) {
  const geo = geoFrom(request);
  if (!geo) return json({ ok: false, reason: 'no-geo' });

  // per-IP rate limit (edge cache, per-colo) — stops bulk fake-visitor spam;
  // legit repeat beacons are already no-ops via the dedup index
  const ipForLimit = request.headers.get('CF-Connecting-IP') || 'unknown';
  const rlKey = new Request('https://rl.hfx3d-visits.internal/' + encodeURIComponent(ipForLimit));
  if (await caches.default.match(rlKey)) {
    return json({ ok: false, reason: 'rate-limited', you: geo }, undefined, 429);
  }

  // dedup keys on the page's localStorage UUID rather than IP — residential
  // IPs rotate and geolocate to different nearby cities
  let cid = null;
  try {
    const body = await request.json();
    if (body && typeof body.cid === 'string' && /^[0-9a-f-]{36}$/i.test(body.cid)) cid = body.cid;
  } catch (e) {}

  const ip = request.headers.get('CF-Connecting-IP') || '';
  const ua = request.headers.get('User-Agent') || '';
  const seed = cid ? 'cid|' + cid : 'ip|' + ip + '|' + ua;
  const hash = await sha256Hex(seed + '|' + env.HASH_SALT);

  const res = await fetch(
    env.SUPABASE_URL + '/rest/v1/visitors?on_conflict=visitor_hash',
    {
      method: 'POST',
      headers: sbHeaders(env, { Prefer: 'resolution=ignore-duplicates,return=minimal' }),
      body: JSON.stringify({
        lat: Math.round(geo.lat * 100) / 100, // ~1 km precision — city level is all we need
        lon: Math.round(geo.lon * 100) / 100,
        city: geo.city,
        country: geo.country,
        visitor_hash: hash,
      }),
    }
  );

  if (res.ok) {
    ctx.waitUntil(
      caches.default.put(
        rlKey,
        new Response('1', { headers: { 'Cache-Control': 'public, max-age=600' } })
      )
    );
  }
  return json({ ok: res.ok, you: geo });
}

async function handleVisitors(request, env, ctx) {
  const cache = caches.default;
  const cacheKey = new Request(new URL(request.url).origin + '/visitors');
  const cached = await cache.match(cacheKey);
  if (cached) return cached;

  const res = await fetch(
    env.SUPABASE_URL + '/rest/v1/visitors?select=lat,lon,city,country',
    { headers: sbHeaders(env) }
  );
  const rows = res.ok ? await res.json() : [];

  const visitors = rows.filter(
    (r) =>
      typeof r.lat === 'number' && typeof r.lon === 'number' &&
      !(r.lat === 0 && r.lon === 0)
  );

  const body = json({ total: visitors.length, visitors }, { 'Cache-Control': 'public, max-age=300' });
  ctx.waitUntil(cache.put(cacheKey, body.clone()));
  return body;
}

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });

    const path = new URL(request.url).pathname;
    if (path === '/visit' && request.method === 'POST') return handleVisit(request, env, ctx);
    if (path === '/visitors' && request.method === 'GET') return handleVisitors(request, env, ctx);
    return json({ error: 'not found' }, undefined, 404);
  },
};
