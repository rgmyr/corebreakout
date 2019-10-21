conda env create -f conda_environment.yml
conda init bash
conda activate corebreakout
cd Mask_RCNN; pip install -e .
cd ..; pip install -e .
