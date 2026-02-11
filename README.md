# EBA-3530

This repository contains various materials used throughout the course. You should check this repository regularly, as materials will be added as we cover them in the course.

---

## Virtual environments
Virtual environments is a practical way to run different applications without risking library dependencies between them. Here are the basic commands in `conda`:

Create a virtual environment using the specifications in `environment.yml`
```bash
conda env create -f environment.yml
``` 

Activate a virtual environment
```bash
conda activate <env name> 
``` 

Deactivate a virtual environment
```bash
conda deactivate
``` 

Delete a virtual environment
```bash
conda env remove --name <env name>
``` 
