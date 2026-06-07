'use strict';

/* ── Navbar ── */
const navbar = document.querySelector('.navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 10);
}, { passive: true });

/* ── Mobile nav ── */
const burger   = document.querySelector('.nav-burger');
const navMobile = document.getElementById('navMobile');
if (burger && navMobile) {
  burger.addEventListener('click', () => navMobile.classList.toggle('open'));
  document.addEventListener('click', e => {
    if (!burger.contains(e.target) && !navMobile.contains(e.target))
      navMobile.classList.remove('open');
  });
}

/* ── Fade-in on scroll ── */
const fadeObs = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); fadeObs.unobserve(e.target); } });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
document.querySelectorAll('.fade-up').forEach(el => fadeObs.observe(el));

/* ── Citation copy ── */
window.copyCitation = function(btn) {
  navigator.clipboard.writeText(document.getElementById('citationText').textContent).then(() => {
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    setTimeout(() => { btn.innerHTML = '<i class="fas fa-copy"></i> Copy'; }, 2000);
  });
};

/* ── Scroll to top ── */
const scrollTop = document.querySelector('.scroll-top');
if (scrollTop) {
  window.addEventListener('scroll', () => scrollTop.classList.toggle('visible', window.scrollY > 400), { passive: true });
  scrollTop.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── Class taxonomy chips ── */
const CLASSES = [
  { name:'Wall',       hex:'#FF2828' }, { name:'Window',    hex:'#00CCFF' },
  { name:'Door',       hex:'#005449' }, { name:'Balcony',   hex:'#FFEE00' },
  { name:'Vegetation', hex:'#00FF50' }, { name:'Stairs',    hex:'#FF99FF' },
  { name:'Terrain',    hex:'#B84600' }, { name:'Roof',      hex:'#FF00B4' },
  { name:'Blinds',     hex:'#7800FF' }, { name:'Other',     hex:'#A0A0A0' },
  { name:'Column',     hex:'#4F0004' }, { name:'Arch',      hex:'#000000' },
];

const taxEl = document.getElementById('classTaxonomy');
if (taxEl) CLASSES.forEach(c => {
  taxEl.insertAdjacentHTML('beforeend',
    `<div class="class-chip">
       <span class="class-dot" style="background:${c.hex};box-shadow:0 0 8px ${c.hex}55"></span>
       ${c.name}
     </div>`);
});
