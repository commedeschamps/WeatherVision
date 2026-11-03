"""Build architecture and paired confusion figures from the implemented model/results."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from src.utils import LABELS
from src.plots import save


def architecture():
    # Dimensions and block counts match torchvision.models.convnext_tiny.
    fig,ax=plt.subplots(figsize=(10,3.25))
    ax.set(xlim=(0,10),ylim=(0,3.25)); ax.axis('off')
    boxes=[(.05,'RGB input','3 × 224 × 224'),(1.65,'4×4 stem / s4','96 × 56 × 56'),(3.25,'Stage 1 / 3 blocks','96 × 56 × 56'),(4.85,'Stage 2 / 3 blocks','192 × 28 × 28'),(6.45,'Stage 3 / 9 blocks','384 × 14 × 14'),(8.05,'Stage 4 / 3 blocks','768 × 7 × 7')]
    for i,(x,title,shape) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,2.28),1.45,.78,boxstyle='round,pad=0.025',fc='#E8F1ED',ec='#55796D',lw=1))
        ax.text(x+.725,2.79,title,ha='center',va='center',fontsize=9.2,weight='bold')
        ax.text(x+.725,2.52,shape,ha='center',va='center',fontsize=9.2)
        if i: ax.annotate('',xy=(x-.03,2.67),xytext=(x-.16,2.67),arrowprops={'arrowstyle':'->','color':'#355F51','lw':1.2})
    ax.text(6.55,2.05,'Between stages: LayerNorm → 2×2 convolution, stride 2',ha='center',fontsize=10)
    ax.annotate('',xy=(8.8,1.78),xytext=(8.8,2.25),arrowprops={'arrowstyle':'->','color':'#355F51'})
    ax.add_patch(FancyBboxPatch((5.85,1.05),3.55,.68,boxstyle='round,pad=0.04',fc='#D8E8DF',ec='#55796D'))
    ax.text(7.62,1.51,'Global average pool → LayerNorm',ha='center',fontsize=10.5)
    ax.text(7.62,1.22,'768 features → Linear(768, 4) → logits',ha='center',fontsize=10.5,weight='bold')
    ax.text(.15,1.65,'Each ConvNeXt block',weight='bold',fontsize=11)
    ax.text(.15,1.31,'7×7 depthwise conv → LayerNorm',fontsize=10.5)
    ax.text(.15,.97,'Linear C→4C → GELU → Linear 4C→C',fontsize=10.5)
    ax.text(.15,.63,'Layer scale → residual addition',fontsize=10.5)
    ax.text(5.9,.65,'Frozen ImageNet backbone and normalization;',fontsize=10)
    ax.text(5.9,.32,'only the 3,076-parameter linear head trains.',fontsize=10)
    fig.tight_layout(pad=.2)
    fig.savefig(ROOT/'outputs/convnext_architecture.svg',bbox_inches='tight')
    save(fig,ROOT/'outputs/convnext_architecture.png')


def paired(runs,titles,name):
    fig,axes=plt.subplots(1,2,figsize=(8,3.05),layout='constrained')
    for ax,run,title in zip(axes,runs,titles):
        matrix=np.array(json.loads((ROOT/f'outputs/{run}/test_metrics.json').read_text())['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=53)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=12,color='white' if matrix[i,j]>29 else '#182B37')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted category',ylabel='True category',title=title)
        ax.tick_params(labelsize=10.5)
    save(fig,ROOT/f'outputs/{name}.png')

if __name__=='__main__':
    architecture()
    paired(['cnn_baseline','mobilenet_improved'],['Simple CNN','MobileNetV2'],'a1_confusion_pair')
    paired(['convnext_aug','convnext_noaug'],['ConvNeXt + augmentation','ConvNeXt without augmentation'],'ablation_confusion_pair')
