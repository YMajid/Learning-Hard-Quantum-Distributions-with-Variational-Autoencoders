# Learning Hard Quantum Distributions with Variational Autoencoders

## Original Paper

You can find the original paper [here](https://www.nature.com/articles/s41534-018-0077-z).

## Project Structure

```
project
├── data
├── param
│   └── parameters.json
├── results
├── src
│   └── model
│       ├── hidden_layers.py
│       ├── model.py
│       └── variational_autoencoder.py
│   └── utils
│       ├── gen_data
│       │   ├── gen_hard.py
│       │   └── gen_random.py
│       ├── create_dataset.py
│       ├── get_data.py
│       └── library.py
└── main.py
```

- `/data`
    - Where the generated data is stored
    - Where the VAE model reads data from
- `/param`
    - Contains the hyperparameters file used
- `/results`
    - Contains loss plots, fidelity plots and saved-trained models
- `/src`
    - `/model`
        - Contains architecture and code to run the model
    - `/utils`
        - Contains the code to generate, save and read datasets
        - `/gen_data`
            - Code to generate the datasets

## Example Command

```
python3 main.py -v 0 -n 18 --param param/parameters.json --pretrained False
```

## Argparse --help

```
usage: main.py [-h] [-v N] [-n N] [--result result/] [--pretrained False]
               [--param param/param.json]

Learning Hard Quantum States Using a Variational AutoEncoder

optional arguments:
  -h, --help            show this help message and exit
  -v N                  Verbosity (0 = all information, else = nothing).
  -n N                  Number of qubits.
  --result result/      Directory for output files.
  --pretrained False    Load pretrained model.
  --param param/param.json
                        Parameter file path.
```

## Contributing

Project (very loosely) follows a [trunk based development style](https://trunkbaseddevelopment.com/).

- Branches are split off from the `master` branch for features, fixes, and all other development.

### Branch Naming

Branches should be prefixed with the following codes to dentore their purpose:

- `feat`: new feature
- `bug`: bug fix
- `chore`: refactoring and misc. tasks
- `docs`: documentation and testing

#### Example

```
docs-making_this_readme
```

### Committing to `master` branch

Make a merge request from your branch to the master branch, once approved it can be merged into the master branch.