# reasoning-augmented-generation

Reasoning Augmented Generation - BH 2024


## Example 
```
cd src
python pipeline.py "proteins that are involved the process of apoptosis"
```
Output

```
The extracted triple is: (bcl-2, involved, apoptosis)

Detected role:  involved -> part of
Detected class:  apoptosis -> Apoptosis modulation

OWL sentence: part of some Apoptosis modulation
```
