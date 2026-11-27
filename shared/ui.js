/* ml-viz shared UI helpers: persisted theme, accessible tabs with #hash deep-linking, reduced-motion check. */
(function (root) {
  'use strict';
  var THEME_KEY = 'ml-viz-theme';

  function getSavedTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (_) { return null; }
  }
  function saveTheme(t) {
    try { localStorage.setItem(THEME_KEY, t); } catch (_) {}
  }
  function systemTheme() {
    return root.matchMedia && root.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') || getSavedTheme() || systemTheme();
  }
  function applyTheme(t, persist) {
    document.documentElement.setAttribute('data-theme', t);
    if (persist) saveTheme(t);
    var btn = document.getElementById('themeBtn');
    if (btn) {
      btn.setAttribute('aria-pressed', String(t === 'dark'));
      btn.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      btn.textContent = t === 'dark' ? 'Theme: dark' : 'Theme: light';
    }
  }

  // Apply persisted theme immediately to avoid flash across pages.
  var initial = getSavedTheme();
  if (initial === 'dark' || initial === 'light') {
    document.documentElement.setAttribute('data-theme', initial);
  }

  function initTheme(btnId) {
    var btn = document.getElementById(btnId || 'themeBtn');
    if (!btn) return;
    applyTheme(currentTheme(), false);
    btn.onclick = function () {
      applyTheme(currentTheme() === 'dark' ? 'light' : 'dark', true);
    };
  }

  function initTabs(opts) {
    opts = opts || {};
    var tablist = document.getElementById(opts.tablistId || 'tabs');
    if (!tablist) return null;
    var tabs = Array.prototype.slice.call(tablist.querySelectorAll('.tab[role="tab"]'));
    var slugs = opts.slugs || {};
    var slugToPanel = {};
    tabs.forEach(function (b) {
      var pid = b.dataset.p;
      if (!b.id) b.id = 'tab-' + pid;
      b.setAttribute('aria-controls', pid);
      var panel = document.getElementById(pid);
      if (panel) {
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', b.id);
      }
      var slug = slugs[pid] || pid;
      slugToPanel[slug] = pid;
      slugToPanel[pid] = pid;
    });

    function activate(pid, updateHash, focusTab) {
      var target = tabs.find(function (b) { return b.dataset.p === pid; }) || tabs[0];
      if (!target) return;
      pid = target.dataset.p;
      tabs.forEach(function (b) {
        var on = b === target;
        b.setAttribute('aria-selected', String(on));
        b.setAttribute('tabindex', on ? '0' : '-1');
        var panel = document.getElementById(b.dataset.p);
        if (panel) panel.hidden = !on;
      });
      if (focusTab) target.focus();
      if (updateHash && root.history && root.history.replaceState) {
        var slug = slugs[pid] || pid;
        root.history.replaceState(null, '', '#' + slug);
      }
      if (opts.onSelect) opts.onSelect(pid);
    }

    tabs.forEach(function (b) {
      b.onclick = function () { activate(b.dataset.p, true, false); };
    });

    tablist.addEventListener('keydown', function (e) {
      var idx = tabs.indexOf(document.activeElement);
      if (idx < 0) return;
      var next = null;
      if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = tabs.length - 1;
      if (next !== null) {
        e.preventDefault();
        activate(tabs[next].dataset.p, true, true);
      }
    });

    function fromHash() {
      var h = (root.location.hash || '').replace(/^#/, '').trim();
      if (h.indexOf('tab=') === 0) h = h.slice(4);
      return slugToPanel[h] || null;
    }

    root.addEventListener('hashchange', function () {
      var pid = fromHash();
      if (pid) activate(pid, false, false);
    });

    var start = fromHash() || (tabs.find(function (b) { return b.getAttribute('aria-selected') === 'true'; }) || tabs[0]).dataset.p;
    activate(start, false, false);
    return { activate: activate };
  }

  function reducedMotion() {
    return Boolean(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  root.mlViz = {
    initTheme: initTheme,
    initTabs: initTabs,
    reducedMotion: reducedMotion
  };
})(typeof window !== 'undefined' ? window : globalThis);
