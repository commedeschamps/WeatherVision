"""Execute the submission notebook locally without retraining completed runs."""
from pathlib import Path
import os,sys,time,json
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
os.environ['JUPYTER_PLATFORM_DIRS']='1'
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter-runtime')
os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython')
notebook=nbformat.read(ROOT/'notebooks/WeatherVision_Colab.ipynb',as_version=4)
client=NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
# Explicit local kernel command avoids writing/installing a user kernelspec.
from jupyter_client import KernelManager
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client.km=km
start=time.time()
client.execute()
notebook.metadata['weathervision_validation']={'execution':'local nbclient/ipykernel','hosted_colab_executed':False,'python':sys.version.split()[0],'code_cells':sum(c.cell_type=='code' for c in notebook.cells),'elapsed_seconds':round(time.time()-start,2),'date':'2026-11-03'}
nbformat.write(notebook,ROOT/'notebooks/WeatherVision_Colab.ipynb')
(ROOT/'outputs/notebook_execution.json').write_text(json.dumps(dict(notebook.metadata['weathervision_validation']),indent=2)+'\n')
print(json.dumps(dict(notebook.metadata['weathervision_validation']),indent=2))
