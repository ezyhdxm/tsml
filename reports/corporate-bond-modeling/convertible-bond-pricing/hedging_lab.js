/* Frozen synthetic results, not live quotes. No new simulation in the browser. */
(() => {
  'use strict';
  const data = @@HEDGING_RESULTS@@;
  const byId = id => document.getElementById(id);
  const colors = ['#146a78', '#b76434', '#6751a5'];
  const dashes = [[], [8, 5], [2, 5]];
  function scoreRows(mode, spreadBp, alpha) {
    if (!['constant','revelation'].includes(mode) || !Number.isFinite(spreadBp) || spreadBp < 0 || !Number.isFinite(alpha) || alpha < 0) {
      throw new Error('Invalid hedging-lab controls');
    }
    const scale = spreadBp / (data.parameters.half_spread * 10000);
    return data[mode].rows.map(row => ({
      steps: row.steps, sd: row.gross_pv.sd,
      cost: row.mean_cost_pv * scale,
      score: row.mean_cost_pv * scale + alpha * row.gross_pv.sd
    }));
  }
  function chart(id, series, yLabel) {
    const canvas = byId(id);
    if (!canvas) return;
    const width = Math.max(240, canvas.getBoundingClientRect().width);
    const height = width < 500 ? 270 : 320;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.height = height + 'px';
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);
    const left = 48, right = 18, top = 36, bottom = 44;
    const innerW = width - left - right, innerH = height - top - bottom;
    const maxY = Math.max(0.1, ...series.flatMap(s => s.values.map(v => v[1]))) * 1.13;
    const x = n => left + (Math.log2(n)-4)/6 * innerW;
    const y = value => top + innerH * (1-value/maxY);
    ctx.font = '12px system-ui, sans-serif';
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#dbe4e8';
    ctx.fillStyle = '#48606b';
    ctx.setLineDash([]);
    for (let i=0; i<=4; i++) {
      const value = maxY*i/4;
      ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(width-right, y(value)); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText(value.toFixed(2), left-7, y(value)+4);
    }
    for (const n of data.parameters.grids) {
      ctx.textAlign = 'center'; ctx.fillText(String(n), x(n), height-bottom+20);
    }
    ctx.textAlign = 'left'; ctx.fillText(yLabel, left, 18);
    ctx.textAlign = 'center'; ctx.fillText('一年内调仓区间数 N（对数刻度）', left+innerW/2, height-5);
    series.forEach((s, index) => {
      ctx.strokeStyle = colors[index % colors.length];
      ctx.fillStyle = colors[index % colors.length];
      ctx.lineWidth = 2.4; ctx.setLineDash(dashes[index % dashes.length]);
      ctx.beginPath();
      s.values.forEach(([n, value], j) => j ? ctx.lineTo(x(n), y(value)) : ctx.moveTo(x(n), y(value)));
      ctx.stroke(); ctx.setLineDash([]);
      for (const [n, value] of s.values) {ctx.beginPath(); ctx.arc(x(n), y(value), 3, 0, 2*Math.PI); ctx.fill();}
    });
  }
  function update() {
    const spread = Number(byId('bsSpread').value);
    const alpha = Number(byId('bsAlpha').value);
    const mode = byId('bsMode').value;
    const rows = scoreRows(mode, spread, alpha);
    const best = rows.reduce((a,b) => a.score <= b.score ? a : b);
    byId('bsSpreadOut').textContent = String(spread);
    byId('bsAlphaOut').textContent = alpha.toFixed(1);
    byId('bsBestGrid').textContent = String(best.steps);
    byId('bsBestScore').textContent = best.score.toFixed(4);
    chart('bsCostChart', [
      {values: rows.map(r => [r.steps, r.sd])},
      {values: rows.map(r => [r.steps, r.cost])},
      {values: rows.map(r => [r.steps, r.score])}
    ], '每份期权 · 折现至今天的金额');
    chart('bsVolRiskChart', [
      {values: data.constant.rows.map(r => [r.steps, r.gross_pv.sd])},
      {values: data.revelation.rows.map(r => [r.steps, r.gross_pv.sd])},
      {values: data.parameters.grids.map(n => [n, data.revelation.continuous_limit_sd_quadrature])}
    ], '无交易费 hedge P&L 的标准差');
  }
  for (const id of ['bsSpread','bsAlpha','bsMode']) byId(id).addEventListener('input', update);
  window.addEventListener('resize', update);
  window.BSHedgingTeaching = {scoreRows, data};
  update();
})();
