from pathlib import Path
import sys,json
r=Path(__file__).parent;sys.path.insert(0,str(r/'.deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
plt.rcParams.update({'font.size':8,'font.family':'STIXGeneral','mathtext.fontset':'stix'})
assets=r.parent/'docs'/'assets';out=r/'figures';out.mkdir(exist_ok=True)
for id,name in [('official-cornell','cbox'),('metal-040','metal'),('sponza-architecture','sponza')]:
 d=assets/id
 fig,axes=plt.subplots(1,3,figsize=(7.1,2.25),layout='constrained')
 for ax,fn,title in zip(axes,['A_reference.png','B_prediction.png','error.png'],['A  Cycles','B  RenderFormer','Mean absolute RGB error']):
  ax.imshow(Image.open(d/fn));ax.axis('off');ax.set_title(title,fontsize=9,pad=4)
 cb=fig.colorbar(ScalarMappable(norm=Normalize(0,.25),cmap='magma'),ax=axes[2],fraction=.05,pad=.025,ticks=[0,.125,.25]);cb.ax.tick_params(labelsize=7)
 fig.savefig(out/(name+'-comparison.png'),dpi=350);plt.close(fig)
data=json.loads((r.parent/'public-data/teapot-parameter_responses.json').read_text());rows={x['id']:x for x in data['rows']}
fig,axs=plt.subplots(1,2,figsize=(7.1,2.45),layout='constrained')
for ax,ids,xkey,ykey,xlabel,ylabel in [(axs[0],['roughness_015','baseline','roughness_075'],'roughness','teapot_p99_Y',r'Roughness $r$',r'Teapot ROI $P_{99}(Y)$'),(axs[1],['wall_000','wall_050','baseline'],'red_wall_strength','floor_red_fraction',r'Red-wall strength $s$',r'Floor ROI $R/(R+G+B)$')]:
 for ab,mk,col in [('A','o-','#225ea8'),('B','s--','#d95f0e')]:
  ax.plot([rows[k][xkey] for k in ids],[rows[k][ab+'_'+ykey] for k in ids],mk,color=col,label=ab+'  '+('Cycles' if ab=='A' else 'RenderFormer'),markersize=4)
 ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(out/'parameter-responses.png',dpi=350);plt.close(fig)
print('four real-result figures generated')
