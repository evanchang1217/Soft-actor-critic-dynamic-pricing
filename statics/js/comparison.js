"use strict";
(() => {
  const el = id => document.getElementById(id);
  const colors = ["#3565a6", "#bd6135", "#8b5ca6"];
  let data = null, timer = null;
  function pause() { if (timer !== null) clearInterval(timer); timer = null; el("play").textContent = "Play"; }
  function chart(id, field, scale=1, reference=null) {
    const canvas = el(id), ctx = canvas.getContext("2d"), width = canvas.clientWidth, height = 230;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio); canvas.height = height * ratio; ctx.scale(ratio, ratio);
    const margin = {left:55, right:15, top:15, bottom:28}, end = Number(el("playhead").value);
    const values = data.controllers.flatMap(c => c.steps.map(s => s[field]*scale));
    let low = Math.min(...values), high = Math.max(...values);
    if (reference !== null) {low = Math.min(low,reference); high = Math.max(high,reference);}
    const pad = (high-low)*.08 || Math.max(Math.abs(high)*.05,.01); low -= pad; high += pad;
    const x = step => margin.left+(step-1)/Math.max(1,data.steps-1)*(width-margin.left-margin.right);
    const y = v => height-margin.bottom-(v-low)/(high-low)*(height-margin.top-margin.bottom);
    ctx.font = "11px system-ui"; ctx.fillStyle = "#657570";
    for(let i=0;i<5;i++) {const v=low+(high-low)*i/4; ctx.strokeStyle="#e7ece7"; ctx.beginPath();ctx.moveTo(margin.left,y(v));ctx.lineTo(width-margin.right,y(v));ctx.stroke();ctx.fillText(v.toLocaleString(undefined,{maximumFractionDigits:2}),2,y(v)+4);}
    if(reference!==null){ctx.setLineDash([4,4]);ctx.strokeStyle="#83918b";ctx.beginPath();ctx.moveTo(margin.left,y(reference));ctx.lineTo(width-margin.right,y(reference));ctx.stroke();ctx.setLineDash([]);}
    data.controllers.forEach((c,i)=>{ctx.strokeStyle=colors[i];ctx.lineWidth=1.7;ctx.beginPath();c.steps.slice(0,end).forEach((s,j)=>{if(j===0)ctx.moveTo(x(s.step),y(s[field]*scale));else ctx.lineTo(x(s.step),y(s[field]*scale));});ctx.stroke();const s=c.steps[end-1];ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x(s.step),y(s[field]*scale),3,0,2*Math.PI);ctx.fill();});
    ctx.fillStyle="#657570";ctx.fillText("1",margin.left,height-7);ctx.fillText(`Step ${data.steps}`,width-85,height-7);
  }
  function render() {
    if(!data)return;
    el("step-value").textContent=el("playhead").value;
    chart("gold-chart","gold_ratio",1,1);chart("volume-chart","volume_per_step");
    chart("low-chart","low_gold_fraction",100);chart("error-chart","mean_gold_error",100,0);
    chart("buy-chart",`buy_${el("price-product").value}`);chart("sell-chart",`sell_${el("price-product").value}`);
  }
  function row(name,s,batch) {
    const tr=document.createElement("tr"); if(batch)tr.className="batch";
    const values=[name,s.score_per_step.toFixed(4),`${(s.mean_gold_error*100).toFixed(2)}%`,`${(s.max_gold_error*100).toFixed(2)}%`,s.volume_per_step.toFixed(0),`${(s.mean_low_gold_fraction*100).toFixed(2)}%`];
    values.forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.appendChild(td);});return tr;
  }
  el("comparison-form").addEventListener("submit",async event=>{
    event.preventDefault();pause();el("load-comparison").disabled=true;
    el("comparison-status").textContent="Loading saved policies and running frozen inference…";
    try {
      const query=new URLSearchParams(new FormData(event.currentTarget));
      const response=await fetch(`${event.currentTarget.action}?${query}`,{cache:"no-store"});
      const result=await response.json();if(!response.ok)throw new Error(result.error || "Comparison could not be loaded.");
      data=result;el("active-comparison").textContent=data.label;
      el("active-seeds").textContent=`Market ${data.market_seed} · SAC seed ${data.model_seed} · ${data.training_steps.toLocaleString()} training steps · fixed 5% fee`;
      el("comparison-context").textContent=data.scenario==="inventory"?"SAC was trained in the consumption + satiation market; P and PI were tuned there too. This setting reverses the batch score ranking seen in the original market.":"Controllers share initial market conditions and resource randomness. Their trade paths differ as their decisions change the market.";
      el("price-product").replaceChildren(...data.products.map((name,i)=>{const o=document.createElement("option");o.value=i;o.textContent=name;return o;}));
      el("comparison-metrics").replaceChildren(...data.controllers.flatMap(c=>[row(`${c.name} · this episode`,c.summary,false),row(`${c.name} · study average`,c.batch,true)]));
      el("batch-note").textContent=`Study averages use ${data.batch_markets} markets and ${data.batch_models} SAC models; P/PI each run once per market. Batch peak error is the mean of episode maxima. These are descriptive means; intervals are in the saved report.`;
      el("record-details").textContent=`Source: ${data.record}. Single-episode illustrations do not replace the multi-seed comparison.`;
      el("model-details").textContent=JSON.stringify({model_sha256:data.model_sha256,protocol_sha256:data.protocol_sha256,behavior:data.behavior,controllers:data.controllers.map(c=>({name:c.name,parameters:c.parameters}))},null,2);
      el("playhead").max=data.steps;el("playhead").value=data.steps;el("comparison-results").hidden=false;render();
      el("comparison-status").textContent="Comparison ready. View the complete trajectories, or press Play to replay from step 1.";
    } catch(error) {el("comparison-status").textContent=error.message;} finally {el("load-comparison").disabled=false;}
  });
  function play(){if(!data)return;if(timer!==null){pause();return;}if(Number(el("playhead").value)>=data.steps)el("playhead").value=1;el("play").textContent="Pause";timer=setInterval(()=>{el("playhead").value=Math.min(data.steps,Number(el("playhead").value)+1);render();if(Number(el("playhead").value)>=data.steps)pause();},1000/Number(el("speed").value));}
  el("play").addEventListener("click",play);
  el("restart").addEventListener("click",()=>{pause();el("playhead").value=1;render();});
  el("playhead").addEventListener("input",()=>{pause();render();});
  el("speed").addEventListener("change",()=>{if(timer!==null){pause();play();}});
  el("price-product").addEventListener("change",render);
  let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);});
  function download(text,ext,type){if(!data)return;const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement("a");a.href=url;a.download=`comparison-${data.scenario}-${data.market_seed}-model-${data.model_seed}.${ext}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  el("download-json").addEventListener("click",()=>{if(data)download(JSON.stringify(data,null,2),"json","application/json");});
  el("download-csv").addEventListener("click",()=>{if(!data)return;const keys=Object.keys(data.controllers[0].steps[0]);const lines=[["scenario","market_seed","model_seed","controller",...keys].join(",")];data.controllers.forEach(c=>c.steps.forEach(s=>lines.push([data.scenario,data.market_seed,data.model_seed,c.name,...keys.map(k=>s[k])].join(","))));download(lines.join("\n")+"\n","csv","text/csv");});
})();
