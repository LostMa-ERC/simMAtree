

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
    
    def get_pymc_priors(self, model=None):
        """Définit et retourne les priors du modèle"""
        pass
