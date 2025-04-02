#!/bin/bash
# -------------------------------------------

# Définir les chemins communs
MODEL_CONFIG_YULE="params/yule.json"
MODEL_CONFIG_BDP="params/birth_death_poisson.json"
INFERENCE_CONFIG_PYMC="params/pymc_config.json"
INFERENCE_CONFIG_SBI="params/sbi_config.json"
OUTPUT_DIR="results/"
DATA_DIR="data/"

# Fonction pour afficher les commandes disponibles
print_help() {
    echo "==== Commandes disponibles pour les tests ====="
    echo "1. generate_yule: Générer des données synthétiques avec le modèle Yule"
    echo "2. generate_bdp: Générer des données synthétiques avec le modèle BirthDeathPoisson"
    echo "3. inference_yule_pymc: Inférence sur les données Yule avec PyMC"
    echo "4. inference_bdp_pymc: Inférence sur les données BDP avec PyMC"
    echo "5. inference_yule_sbi: Inférence sur les données Yule avec SBI"
    echo "6. run_quick_test: Exécuter un test rapide (génération et inférence avec peu d'échantillons)"
    echo "Usage: ./test_commands.sh [commande]"
}

# Génération de données synthétiques avec le modèle Yule
generate_yule() {
    echo "Génération de données avec le modèle Yule..."
    python run.py --task generate \
                 --data_path "${DATA_DIR}synthetic_yule.csv" \
                 --model_config "${MODEL_CONFIG_YULE}"
}

# Génération de données synthétiques avec le modèle BirthDeathPoisson
generate_bdp() {
    echo "Génération de données avec le modèle BirthDeathPoisson..."
    python run.py --task generate \
                 --data_path "${DATA_DIR}synthetic_bdp.csv" \
                 --model_config "${MODEL_CONFIG_BDP}"
}

# Inférence avec le modèle Yule et PyMC
inference_yule_pymc() {
    echo "Inférence avec le modèle Yule et PyMC..."
    python run.py --task inference \
                 --data_path "${DATA_DIR}simulation.csv" \
                 --model_config "${MODEL_CONFIG_YULE}" \
                 --inference_config "${INFERENCE_CONFIG_PYMC}" \
                 --results_dir "${OUTPUT_DIR}yule_pymc/"
}

# Inférence avec le modèle BirthDeathPoisson et PyMC
inference_bdp_pymc() {
    echo "Inférence avec le modèle BirthDeathPoisson et PyMC..."
    python run.py --task inference \
                 --data_path "${DATA_DIR}simulation.csv" \
                 --model_config "${MODEL_CONFIG_BDP}" \
                 --inference_config "${INFERENCE_CONFIG_PYMC}" \
                 --results_dir "${OUTPUT_DIR}bdp_pymc/"
}

# Inférence avec le modèle Yule et SBI
inference_yule_sbi() {
    echo "Inférence avec le modèle Yule et SBI..."
    python run.py --task inference \
                 --data_path "${DATA_DIR}synthetic_yule.csv" \
                 --model_config "${MODEL_CONFIG_YULE}" \
                 --inference_config "${INFERENCE_CONFIG_SBI}" \
                 --results_dir "${OUTPUT_DIR}yule_sbi/"
}

# Test rapide (peu d'échantillons pour un test rapide)
run_quick_test() {
    echo "Exécution d'un test rapide..."
    
    python run.py --task inference \
                 --data_path "${DATA_DIR}simulation.csv" \
                 --model_config "${MODEL_CONFIG_YULE}" \
                 --inference_config "${INFERENCE_CONFIG_SBI}" \
                 --results_dir "${OUTPUT_DIR}yule_sbi/"
}

# Exécuter la commande sélectionnée
if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

case "$1" in
    "generate_yule")
        generate_yule
        ;;
    "generate_bdp")
        generate_bdp
        ;;
    "inference_yule_pymc")
        inference_yule_pymc
        ;;
    "inference_bdp_pymc")
        inference_bdp_pymc
        ;;
    "inference_yule_sbi")
        inference_yule_sbi
        ;;
    "run_quick_test")
        run_quick_test
        ;;
    *)
        echo "Commande inconnue: $1"
        print_help
        ;;
esac