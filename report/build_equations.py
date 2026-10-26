from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/".deps"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset']='stix'
plt.rcParams['font.family']='STIXGeneral'
root=Path(__file__).parent/'equations';root.mkdir(exist_ok=True)
import json
items=json.loads((Path(__file__).parent/'equations.json').read_text(encoding='utf-8'))
for name,latex in items.items():
 fig=plt.figure(figsize=(7,1));fig.text(.5,.5,'$'+latex+'$',ha='center',va='center',fontsize=18);fig.savefig(root/(name+'.png'),dpi=400,bbox_inches='tight',pad_inches=.06,transparent=True);plt.close(fig)
print('four math figures created')
