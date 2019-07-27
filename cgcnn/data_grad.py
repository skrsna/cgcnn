from __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from sklearn.preprocessing import OneHotEncoder
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.base import TransformerMixin
import mongo
import copy

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    distances: torch.Tensor shape (N, 1)
      Storing connectivity information of atoms
    connection_atom_idx: torch.Tensor shape (N, 1)
      One hot encoding representation of the connectivity
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_nbr_fea_offset = [], [], [], []
    batch_fixed_atom_idx, batch_atom_pos, batch_nbr_pos = [], [], []
    crystal_atom_idx, crystal_fixed_atom_idx, batch_target = [], [], []
    crystal_cell, crystal_cell_idx = [], []
    fixed_atom_mask, batch_atom_pos_idx, batch_atom_pos_final = [], [], []
    base_idx = 0
    cell_idx = 0
    base_fixed_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, nbr_fea_offset, atom_pos, nbr_pos, atom_pos_idx, cells, fixed_atom_idx, atom_pos_final), target)\
            in enumerate(dataset_list):
            
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        
#         batch_fixed_atom_idx.append(fixed_atom_idx)
#         new_fixed_idx = torch.LongTensor(np.arange(len(fixed_atom_idx))+base_fixed_idx)
#         crystal_fixed_atom_idx.append(new_fixed_idx)
#         base_fixed_idx += len(fixed_atom_idx)
        
        batch_atom_pos.append(atom_pos)
        batch_nbr_pos.append(nbr_pos)
    
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_offset.append(nbr_fea_offset)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        
        crystal_cell.append(cells)
        crystal_cell_idx.append(cell_idx)

        fixed_base = np.zeros((n_i, 1))
        fixed_base[fixed_atom_idx] = 1
        fixed_atom_mask.append(torch.FloatTensor(fixed_base))
#         fixed_atom_mask.append(torch.FloatTensor(fixed_atom_idx))
        free_atom_idx = np.where(fixed_base == 0)[0]

        batch_atom_pos_idx.append(atom_pos_idx + base_idx)
        
#         if type(target) != float:
#             batch_target.append(torch.Tensor(target[0]))
#         else:
#             batch_target.append(torch.Tensor(0))
        if type(target) is not float:
            batch_target.append(torch.Tensor(target[0][free_atom_idx]))
        else:
            batch_target.append(torch.Tensor([0]))

        batch_atom_pos_final.append(atom_pos_final[free_atom_idx])
    
        cell_idx += 1
        base_idx += n_i
#     print('atom', torch.cat(batch_atom_fea,dim=0).shape)
#     print('target', torch.cat(batch_target).shape)
    return {'atom_fea':torch.cat(batch_atom_fea, dim=0), 
            'nbr_fea':torch.cat(batch_nbr_fea, dim=0), 
            'nbr_fea_idx':torch.cat(batch_nbr_fea_idx, dim=0), 
            'nbr_fea_offset':torch.cat(batch_nbr_fea_offset, dim=0),
            'crystal_atom_idx':crystal_atom_idx,
#             'fixed_atom_idx':torch.cat(batch_fixed_atom_idx,dim=0),
#             'crystal_fixed_atom_idx':crystal_fixed_atom_idx,
            'atom_pos':torch.cat(batch_atom_pos, dim=0),
            'nbr_pos':torch.cat(batch_nbr_pos, dim=0),
            'atom_pos_idx': torch.cat(batch_atom_pos_idx, dim=0),
            'cells': torch.cat(crystal_cell, dim=0),
            'fixed_atom_mask': torch.cat(fixed_atom_mask),
            'atom_pos_final': torch.cat(batch_atom_pos_final)}, torch.cat(batch_target)#torch.cat(batch_atom_pos_final) 
# torch.FloatTensor(batch_target) 
# torch.cat(batch_target).view(-1,1)



class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance 
        y

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class StructureData():
    """
    
    RE-COMMENT THIS 
    
    
    Parameters
    ----------

    atoms_list: list of ASE atoms objects
        List of ASE atoms objects (final relaxed geometry)
    atoms_list_initial_config: list of ASE atoms objects
        List of ASE atoms objects (initial unrelaxed geometry)
        This is very important if the model will be used to predict
        the properties of unrelaxed structures
    atom_init_loc: str
        The location of the atom_init.json file that contains atomic properties
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    use_voronoi: bool
        Controls whether the original (pair distance) or voronoi
        method from pymatgen is used to determine neighbor lists 
        and distances.
    use_fixed_info: bool
        If True, add whether each atom is fixed by ASE constraints as an atomic feature.
        Hypothesized to improve the fit because there is information in the fixed
        atoms being in the bulk
    use_tag: 
        If true, add the ASE tag as an atomic feature
    use_distance:
        If true, for each atom add a graph distance from the atom to the nearest atom
        on the graph that has a tag of 1 (indicated it is an adsorbate atom in our scheme). 
        This allows atoms near the adsorbate to have a higher influence if the model
        deems it helpful.
    train_geometry: str
        If 'final', use the final relaxed structure for input to the graph
        If 'initial' use the initial unrelaxed structure
        If 'final-adsorbate', 'use the initial relax structure for everything with tag=0,
            but add a fixed-edge feature to adsorbate atoms in the final configuration.
            We did this so that the information from adsorbate movement (ex. on-top to bridge)
            is included in the input space, but the final relaxed bond distance is not included.
            This makes the method transferable to the predictions for unrelaxed structures with
            various adsorbate locations

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    """
    def __init__(self, atoms_list, atoms_list_initial_config, atom_init_loc, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123, use_voronoi=False, use_fixed_info=False, use_tag=False, use_distance=False, bond_property=True, train_geometry='initial', is_initial=True):
        
        #this copy is very important; otherwise things ran, but there was some sort 
        # of shuffle that was affecting the real list, resulting in weird loss
        # loss functions and poor training
        self.atoms_list = copy.deepcopy(atoms_list)
        self.atoms_list_initial_config = copy.deepcopy(atoms_list_initial_config)
        
        self.atom_init_loc = atom_init_loc
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.use_voronoi = use_voronoi
        
        #Load the atom features and gaussian distribution functions
        assert os.path.exists(self.atom_init_loc), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_loc)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        
        #Store some tags inside the object for later use
        self.use_fixed_info = use_fixed_info
        self.use_tag = use_tag
        self.use_distance = use_distance
        self.train_geometry = train_geometry #could be initial, final, or final-adsorbate?
        self.bond_property = bond_property
        self.is_initial = is_initial
        
    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
#         atoms = copy.deepcopy(self.atoms_list[idx])
#         atoms_initial_config = copy.deepcopy(self.atoms_list_initial_config[idx])
        
        if self.is_initial:
            atoms = copy.deepcopy(self.atoms_list_initial_config[idx])
            atoms_final = copy.deepcopy(self.atoms_list[idx])
            
            crystal = AseAtomsAdaptor.get_structure(atoms)
            crystal_final = AseAtomsAdaptor.get_structure(atoms_final)
        else:
            atoms = copy.deepcopy(self.atoms_list[idx])
            atoms_initial_config = copy.deepcopy(self.atoms_list_initial_config[idx])
            
            crystal = AseAtomsAdaptor.get_structure(atoms)
            crystal_initial_config = AseAtomsAdaptor.get_structure(atoms_initial_config)
            crystal_final = AseAtomsAdaptor.get_structure(atoms)
        
        #Stack the features from atom_init
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
                              
        # If use_tag=True, then add the tag as an atom feature
        if self.use_tag:
            atom_fea = np.hstack([atom_fea,atoms.get_tags().reshape((-1,1))])
            
        # If use_fixed_info=True, then add whether the atom is fixed by ASE constraint to the features
        if self.use_fixed_info:
            fix_loc, = np.where([type(constraint)==FixAtoms for constraint in atoms.constraints])
            fix_atoms_indices = set(atoms.constraints[fix_loc[0]].get_indices())
            fixed_atoms = np.array([i in fix_atoms_indices for i in range(len(atoms))]).reshape((-1,1))
            atom_fea = np.hstack([atom_fea,fixed_atoms])

        # for bond distance optimization    
        if self.bond_property:
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True, include_image=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea, nbr_fea_offset = [], [], []
            for nbr in all_nbrs:
                assert len(nbr) >= self.max_num_nbr

                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
                nbr_fea_offset.append(list(map(lambda x: x[3], nbr[:self.max_num_nbr])))

            nbr_fea_idx, nbr_fea, nbr_fea_offset = np.array(nbr_fea_idx), np.array(nbr_fea), np.array(nbr_fea_offset)            
            nbr_fea = self.gdf.expand(nbr_fea)
            distances = [0]*len(atoms)

            all_nbrs_final = crystal_final.get_all_neighbors(self.radius, include_index=True, include_image=True)
            all_nbrs_final = [sorted(nbrs_final, key=lambda x: x[1]) for nbrs_final in all_nbrs_final]
            nbr_fea_idx_final, nbr_fea_final, nbr_fea_offset_final = [], [], []

            for nbr_final in all_nbrs_final:
                assert len(nbr_final) >= self.max_num_nbr

                nbr_fea_idx_final.append(list(map(lambda x: x[2],
                                            nbr_final[:self.max_num_nbr])))
                nbr_fea_final.append(list(map(lambda x: x[1],
                                        nbr_final[:self.max_num_nbr])))
                nbr_fea_offset_final.append(list(map(lambda x: x[3], nbr_final[:self.max_num_nbr])))

            nbr_fea_idx_final, nbr_fea_final, nbr_fea_offset_final = np.array(nbr_fea_idx_final), np.array(nbr_fea_final), np.array(nbr_fea_offset_final)            
            nbr_fea_final = self.gdf.expand(nbr_fea_final)
            distances = [0]*len(atoms)
            try:
                nbr_fea = torch.Tensor(nbr_fea)
            except RuntimeError:
                print(nbr_fea)
            

            fixed_atom_idx = atoms.constraints[0].get_indices()
            free_atom_idx = atoms.constraints[0].get_indices
            nbr_pos = atoms.get_positions()[nbr_fea_idx.astype(int)]
            atom_pos = atoms.get_positions()
#             atom_pos = np.tile(pos, (1,self.max_num_nbr)).reshape(len(atoms), self.max_num_nbr, 3)
            atom_pos_idx = np.repeat(np.arange(len(atom_pos)), self.max_num_nbr).reshape(len(atom_pos), self.max_num_nbr)
            cell = atoms.cell
            cells = np.tile(cell, (len(atoms),1)).reshape(len(atoms), 3, 3)
            
            nbr_pos_final = atoms_final.get_positions()[nbr_fea_idx_final.astype(int)]
            atom_pos_final = min_diff(atoms, atoms_final) + atoms.positions
#             atom_pos_final = np.tile(pos_final, (1,self.max_num_nbr)).reshape(len(atoms_final), self.max_num_nbr, 3)
            cell_final = atoms_final.cell
            cells_final = np.tile(cell_final, (len(atoms_final),1)).reshape(len(atoms_final), 3, 3)
            
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        nbr_fea_offset = torch.Tensor(nbr_fea_offset)
        distances=torch.LongTensor(distances)
        atom_fea = torch.Tensor(atom_fea)
        fixed_atom_idx = torch.LongTensor(fixed_atom_idx)
        atom_pos = torch.Tensor(atom_pos)
        nbr_pos = torch.Tensor(nbr_pos)
        atom_pos_idx = torch.LongTensor(atom_pos_idx)
        cells = torch.Tensor(cells)
        
        nbr_fea_idx_final = torch.LongTensor(nbr_fea_idx_final)
        nbr_fea_offset_final = torch.Tensor(nbr_fea_offset_final)
        atom_pos_final = torch.Tensor(atom_pos_final)
        nbr_pos_final = torch.Tensor(nbr_pos_final)
        cells_final = torch.Tensor(cells_final)

        return (atom_fea, nbr_fea, nbr_fea_idx, nbr_fea_offset, atom_pos, nbr_pos, atom_pos_idx, cells, fixed_atom_idx, atom_pos_final)
#                 ,nbr_fea_idx_final, nbr_fea_offset_final, atom_pos_final, nbr_pos_final, cells_final)

class ListDataset():
    def __init__(self, list_in):
        self.list = list_in
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

class StructureDataTransformer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return
    
    def transform(self,X):
        structure_list = [mongo.make_atoms_from_doc(doc) for doc in X] #delete first two adsorbate ATOMS !!!!
        structure_list_orig = [mongo.make_atoms_from_doc(doc['initial_configuration']) for doc in X] #delete first two adsorbate ATOMS !!!!

        SD = StructureData(structure_list, structure_list_orig, *self.args, **self.kwargs)
        return SD

    def fit(self,*_):
        return self

def min_diff(atoms_init, atoms_final):
    positions = (atoms_final.positions-atoms_init.positions)

    fractional = np.linalg.solve(atoms_init.get_cell(complete=True).T,
                                     positions.T).T
    if True:
        for i, periodic in enumerate(atoms_init.pbc):
            if periodic:
                # Yes, we need to do it twice.
                # See the scaled_positions.py test.
                fractional[:, i] %= 1.0
                fractional[:, i] %= 1.0
                
    fractional[fractional>0.5]-=1
    return np.matmul(fractional,atoms_init.get_cell(complete=True).T)



class MergeDataset(torch.utils.data.Dataset):
    #Simple custom dataset to combine two datasets 
    # (one for input X, one for label y)
    def __init__(
            self,
            X,
            y,
            length=None,
    ):

        self.X = X
        self.y = copy.deepcopy(y)

        len_X = len(X)
        if y is not None:
            len_y = len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, i):
        X, y = self.X, self.y
        
        if y is not None:
            yi = copy.deepcopy(y[i])
        else:
            yi = np.nan

        return X[i], yi

def distance_to_adsorbate_feature(atoms, VC, max_dist = 6):    
    # This function looks at an atoms object and attempts to find
    # the minimum distance from each atom to one of the adsorbate 
    # atoms (marked with tag==1)
    conn = copy.deepcopy(VC.connectivity_array)
    conn = np.max(conn,2)

    for i in range(len(conn)):
        conn[i]=conn[i]/np.max(conn[i])

    #get a binary connectivity matrix
    conn=(conn>0.3)*1
    
    #Everything is connected to itself, so add a matrix with zero on the diagonal 
    # and a large number on the off-diagonal
    ident_connection = np.eye(len(conn))
    ident_connection[ident_connection==0]=max_dist+1
    ident_connection[ident_connection==1]=0

    #For each distance, add an array of atoms that can be connected at that distance
    arrays = [ident_connection]
    for i in range(1,max_dist):
        arrays.append((np.linalg.matrix_power(conn,i)>=1)*i+(np.linalg.matrix_power(conn,i)==0)*(max_dist+1))

    #Find the minimum distance from each atom to every other atom (over possible distances)
    arrays=np.min(arrays,0)

    # Find the minimum distance from one of the adsorbate atoms to the other atoms
    min_distance_to_adsorbate = np.min(arrays[atoms.get_tags()==1],0).reshape((-1,1))
    
    #Make sure all of the one hot distance vectors are encoded to the same length. 
    # Encode, return
    min_distance_to_adsorbate[min_distance_to_adsorbate>=max_dist]=max_dist-1
    OHE = OneHotEncoder(categories=[range(max_dist)]).fit(min_distance_to_adsorbate)
    return min_distance_to_adsorbate, OHE.transform(min_distance_to_adsorbate).toarray()

 

