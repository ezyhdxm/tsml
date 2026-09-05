'use strict';
// Standalone teaching calculator. No telemetry, network requests, or market data.
const CB_RESULTS = @@RESULTS_JSON@@;
function normalCDF(x){const a=Math.abs(x)/Math.SQRT2;const t=1/(1+0.3275911*a);const erf=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a);return .5*(1+Math.sign(x)*erf);}
function exactPrice(S=50,sigma=.3,h=.025){const F=100,m=2,T=3,r=.04,q=.01,R=.4,a=r+h;const rec=R*F*h*(-Math.expm1(-a*T))/a;const d1=(Math.log(S/(F/m))+(a-q+.5*sigma*sigma)*T)/(sigma*Math.sqrt(T));const d2=d1-sigma*Math.sqrt(T);return m*S*Math.exp(-q*T)*normalCDF(d1)+F*Math.exp(-a*T)*normalCDF(-d2)+rec;}
function straightPrice(h=.025,coupon=false){const a=.04+h;let v=100*Math.exp(-a*3)+40*h*(-Math.expm1(-a*3))/a;if(coupon)for(let i=1;i<=6;i++)v+=Math.exp(-a*i/2);return v;}
function treePrice(S=50,sigma=.3,h=.025,type='european',N=300){
 if(!(S>0&&sigma>0&&h>=0&&Number.isInteger(N)&&N>0&&N%6===0))throw Error('Invalid inputs or coupon grid');
 const T=3,r=.04,q=.01,R=.4,F=100,m=2,dt=T/N,u=Math.exp(sigma*Math.sqrt(dt)),d=1/u;
 const p=(Math.exp((r+h-q)*dt)-d)/(u-d);if(!(p>=0&&p<=1))throw Error('Transition probability outside [0,1]');
 const disc=Math.exp(-(r+h)*dt),rec=R*F*h*(-Math.expm1(-(r+h)*dt))/(r+h),american=type!=='european';
 const call=type==='call'||type==='both',put=type==='put'||type==='both',coupon=american&&type!=='american_zero',stepCoupon=N/6,logu=Math.log(u);
 let v=new Float64Array(N+1);for(let j=0;j<=N;j++)v[j]=Math.max(F,m*S*Math.exp((2*j-N)*logu))+(coupon?1:0);
 for(let i=N-1;i>=0;i--){for(let j=0;j<=i;j++){
  let continuation=disc*(p*v[j+1]+(1-p)*v[j])+rec;
  const conversion=m*S*Math.exp((2*j-i)*logu);
  if(call&&i>=N/2)continuation=Math.min(continuation,Math.max(105,conversion));
  if(put&&i===N/2)continuation=Math.max(continuation,100);
  if(american)continuation=Math.max(continuation,conversion);
  if(coupon&&i>0&&i%stepCoupon===0)continuation+=1;
  v[j]=continuation;
 }}return v[0];
}
function drawLines(id,series,xlabel,ylabel,logAxes=false,vertical=null){
 const c=document.getElementById(id);if(!c)return;const width=c.clientWidth,height=c.clientHeight,ratio=window.devicePixelRatio||1;c.width=width*ratio;c.height=height*ratio;
 const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
 const pad={l:55,r:17,t:26,b:44},w=width-pad.l-pad.r,hh=height-pad.t-pad.b;
 const tx=x=>logAxes?Math.log10(x):x,ty=y=>logAxes?Math.log10(Math.max(y,1e-10)):y;
 let xs=series.flatMap(s=>s.points.map(p=>tx(p[0]))),ys=series.flatMap(s=>s.points.map(p=>ty(p[1])));
 let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const range=ymax-ymin||1;ymax+=.09*range;ymin-=.06*range;if(!logAxes)ymin=Math.max(0,ymin);
 const X=x=>pad.l+(tx(x)-xmin)/(xmax-xmin)*w,Y=y=>pad.t+(ymax-ty(y))/(ymax-ymin)*hh;
 ctx.font='11px system-ui';ctx.fillStyle='#617383';ctx.strokeStyle='#dbe4eb';ctx.lineWidth=1;
 for(let i=0;i<=4;i++){let y=ymin+(ymax-ymin)*i/4,yy=pad.t+hh*(1-i/4);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(width-pad.r,yy);ctx.stroke();ctx.textAlign='right';ctx.fillText(logAxes?Math.pow(10,y).toFixed(3):y.toFixed(0),pad.l-8,yy+4);}
 let ticks=logAxes?[60,120,300,600,1200,2400]:[0,1,2,3,4,5].map(i=>xmin+(xmax-xmin)*i/5);
 ticks.forEach(x=>{if(tx(x)<xmin||tx(x)>xmax)return;ctx.textAlign='center';ctx.fillText(x.toFixed(0),X(x),height-pad.b+19);});
 ctx.textAlign='left';ctx.fillText(ylabel,8,14);ctx.textAlign='center';ctx.fillText(xlabel,pad.l+w/2,height-5);
 const colors=['#087f8c','#ad7130','#738296','#334e68'];
 series.forEach((s,k)=>{ctx.strokeStyle=colors[k%colors.length];ctx.lineWidth=k===0?2.7:1.7;ctx.setLineDash(s.dash||[]);ctx.beginPath();s.points.forEach((p,j)=>{if(j===0)ctx.moveTo(X(p[0]),Y(p[1]));else ctx.lineTo(X(p[0]),Y(p[1]));});ctx.stroke();});ctx.setLineDash([]);
 if(vertical!==null&&tx(vertical)>=xmin&&tx(vertical)<=xmax){ctx.strokeStyle='#798a99';ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(X(vertical),pad.t);ctx.lineTo(X(vertical),height-pad.b);ctx.stroke();ctx.setLineDash([]);}
}
function shapePlot(){const points=Array.from({length:101},(_,i)=>15+i);drawLines('shapeChart',[{points:points.map(s=>[s,exactPrice(s)])},{points:points.map(s=>[s,2*s]),dash:[7,5]},{points:points.map(s=>[s,straightPrice()]),dash:[2,5]}],'Stock price S','Value / 100 face');}
function convergencePlot(){drawLines('convergenceChart',[{points:CB_RESULTS.convergence.map(x=>[x.N,Math.abs(x.defaultable-CB_RESULTS.analytic.defaultable_european)])},{points:CB_RESULTS.convergence.map(x=>[x.N,Math.abs(x.risk_free-CB_RESULTS.analytic.risk_free_european)]),dash:[7,5]}],'Tree steps N (log scale)','Absolute error (log scale)',true);}
function updateLab(){
 const S=Number(document.getElementById('spotInput').value),sigma=Number(document.getElementById('volInput').value)/100,h=Number(document.getElementById('hazInput').value)/100,type=document.getElementById('contractInput').value;
 document.getElementById('spotOut').textContent=S.toFixed(0);document.getElementById('volOut').textContent=(sigma*100).toFixed(0)+'%';document.getElementById('hazOut').textContent=(h*100).toFixed(1)+'%';
 try{const value=treePrice(S,sigma,h,type);document.getElementById('labPrice').textContent=value.toFixed(4);document.getElementById('labParity').textContent=(2*S).toFixed(4);document.getElementById('labExact').textContent=exactPrice(S,sigma,h).toFixed(4);
 document.getElementById('labNote').textContent=type==='european'?'300 步树与欧式闭式的差值：'+(value-exactPrice(S,sigma,h)).toFixed(5)+'。此处强度改变时，存续股价漂移同步变化。':type==='american_zero'?'300 步、零息、每一网格时点允许转股；欧式闭式值只用于显示对应欧式基准。':'300 步、每半年付息；call 从 1.5 年起立即结算，put 仅在 1.5 年。未模拟 soft call、notice 或真实结算条款。';
 const xs=Array.from({length:36},(_,i)=>15+i*3);drawLines('labChart',[{points:xs.map(s=>[s,treePrice(s,sigma,h,type)])},{points:xs.map(s=>[s,exactPrice(s,sigma,h)]),dash:[7,5]},{points:xs.map(s=>[s,2*s]),dash:[2,5]}],'Stock price S','Value / 100 face',false,S);
 }catch(e){document.getElementById('labNote').textContent=e.message;document.getElementById('labPrice').textContent='输入错误';}
}
let pending=false;function scheduleLab(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;updateLab();});}
['spotInput','volInput','hazInput','contractInput'].forEach(id=>document.getElementById(id).addEventListener('input',scheduleLab));
shapePlot();convergencePlot();updateLab();
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{shapePlot();convergencePlot();updateLab();},120);});
const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){document.querySelectorAll('#sidebar li').forEach(li=>li.classList.remove('active'));const a=document.querySelector('#sidebar a[href="#'+e.target.id+'"]');if(a)a.parentElement.classList.add('active');}});},{rootMargin:'-5% 0px -75% 0px'});document.querySelectorAll('article h2').forEach(h=>observer.observe(h));
window.CBTeaching={exactPrice,treePrice,straightPrice};
