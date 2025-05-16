import torch

class BaseModel:
    """Classe abstraite pour définir l'interface d'un modèle"""
    
    def get_constraints(self, model, params):
        """Définit et retourne les contraintes du modèle"""
        pass
    
    def get_priors(self):
        """Définit et retourne les priors du modèle"""
        pass

    def get_simulator(self):
        """Définit et retourne les priors du modèle"""
        pass

    def validate_params(self, params):
        """Valide et corrige si nécessaire les paramètres avant simulation
        
        Args:
            params: Les paramètres à valider
            
        Returns:
            Les paramètres validés/corrigés
        """
        return params
    
    def get_pymc_priors(self, model=None):
        #TODO : Faire une fonction get_prior global qui utilise soit pymc, soit sbi en fonction d'un argument ? 
        """Définit et retourne les priors du modèle"""
        pass

    def sample_from_prior(self, n_samples=1, device='cpu', param_names = None):

        if not hasattr(self, 'get_sbi_priors'):
            raise NotImplementedError("La méthode get_sbi_priors doit être implémentée")
        
        prior = self.get_sbi_priors(device=device)
        sample_shape = torch.Size([n_samples])
        samples = prior.sample(sample_shape)

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        
        if param_names is None:
            param_names = [f"param_{i}" for i in range(samples.shape[1])]
            
        json_samples = []
        for i in range(n_samples):
            sample_dict = {name: float(value) for name, value in zip(param_names, samples[i])}
            json_samples.append(sample_dict)
        
        return json_samples
