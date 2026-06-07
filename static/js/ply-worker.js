'use strict';
var TS={float:4,double:8,uchar:1,char:1,uint:4,int:4,ushort:2,short:2};
var CC=[[1,.157,.157],[0,.8,1],[0,.329,.286],[1,.933,0],[0,1,.314],[1,.6,1],[.722,.275,0],[1,0,.706],[.471,0,1],[.627,.627,.627],[.310,0,.016],[0,0,0]];
function heat(t){t=Math.max(0,Math.min(1,t));return t<.33?[t*3,0,0]:t<.67?[1,(t-.33)*3,0]:[1,1,(t-.67)*3];}
function jet(t){t=Math.max(0,Math.min(1,t));return[Math.min(1,Math.max(0,1.5-Math.abs(4*t-3))),Math.min(1,Math.max(0,1.5-Math.abs(4*t-2))),Math.min(1,Math.max(0,1.5-Math.abs(4*t-1)))];}
function parsePLY(buf){
  var b=new Uint8Array(buf),tag='end_header\n',he=-1;
  outer:for(var i=0;i<Math.min(b.length,8192)-tag.length;i++){for(var j=0;j<tag.length;j++)if(b[i+j]!==tag.charCodeAt(j))continue outer;he=i+tag.length;break;}
  if(he<0)throw new Error('PLY: no end_header');
  var hdr=new TextDecoder('ascii').decode(b.slice(0,he));
  var nV=0,P=[];
  hdr.split('\n').forEach(function(ln){var p=ln.trim().split(/\s+/);if(p[0]==='element'&&p[1]==='vertex')nV=parseInt(p[2]);else if(p[0]==='property'&&p.length===3&&TS[p[1]])P.push({n:p[2],t:p[1],s:TS[p[1]]});});
  var str=P.reduce(function(s,p){return s+p.s;},0);
  var xyz=new Float32Array(nV*3),rgb=new Float32Array(nV*3),lbl=new Uint8Array(nV),ins=new Float32Array(nV),gps=new Float32Array(nV);
  var hI=false,hG=false;
  P.forEach(function(p){if(p.n==='intensity')hI=true;if(p.n==='gps_time')hG=true;});
  var off=[],r=0;P.forEach(function(p){off.push(r);r+=p.s;});
  var dv=new DataView(buf,he);
  for(var i=0;i<nV;i++){
    var base=i*str;
    for(var pi=0;pi<P.length;pi++){
      var p=P[pi],o=base+off[pi],v=0;
      switch(p.t){case'float':v=dv.getFloat32(o,true);break;case'double':v=dv.getFloat64(o,true);break;case'uchar':v=dv.getUint8(o);break;case'char':v=dv.getInt8(o);break;case'uint':v=dv.getUint32(o,true);break;case'int':v=dv.getInt32(o,true);break;case'ushort':v=dv.getUint16(o,true);break;case'short':v=dv.getInt16(o,true);break;}
      var vi=i*3;
      switch(p.n){case'x':xyz[vi]=v;break;case'y':xyz[vi+1]=v;break;case'z':xyz[vi+2]=v;break;case'red':rgb[vi]=v/255;break;case'green':rgb[vi+1]=v/255;break;case'blue':rgb[vi+2]=v/255;break;case'label':lbl[i]=v;break;case'intensity':ins[i]=v;break;case'gps_time':gps[i]=v;break;}
    }
  }
  return{xyz:xyz,rgb:rgb,lbl:lbl,ins:ins,gps:gps,nV:nV,hI:hI,hG:hG};
}
function semClr(lbl,n){var a=new Float32Array(n*3);for(var i=0;i<n;i++){var c=CC[lbl[i]]||[.5,.5,.5];a[i*3]=c[0];a[i*3+1]=c[1];a[i*3+2]=c[2];}return a;}
function sclClr(v,n,fn){var mn=1e38,mx=-1e38;for(var i=0;i<n;i++){if(v[i]<mn)mn=v[i];if(v[i]>mx)mx=v[i];}var rng=mx>mn?mx-mn:1,a=new Float32Array(n*3);for(var i=0;i<n;i++){var c=fn((v[i]-mn)/rng);a[i*3]=c[0];a[i*3+1]=c[1];a[i*3+2]=c[2];}return a;}
self.onmessage=function(e){
  var url=e.data.url,id=e.data.id;
  fetch(url).then(function(r){if(!r.ok)throw new Error('HTTP '+r.status);return r.arrayBuffer();}).then(function(buf){
    var d=parsePLY(buf);
    var sem=semClr(d.lbl,d.nV);
    var intC=d.hI?sclClr(d.ins,d.nV,heat):d.rgb.slice();
    var timC=d.hG?sclClr(d.gps,d.nV,jet):d.rgb.slice();
    var lblSet={};for(var i=0;i<d.nV;i++)lblSet[d.lbl[i]]=1;
    var pres=Object.keys(lblSet).map(Number);
    self.postMessage({id:id,xyz:d.xyz,rgb:d.rgb,sem:sem,intC:intC,timC:timC,nV:d.nV,hI:d.hI,hG:d.hG,pres:pres},[d.xyz.buffer,d.rgb.buffer,sem.buffer,intC.buffer,timC.buffer]);
  }).catch(function(err){self.postMessage({id:id,error:err.message});});
};
