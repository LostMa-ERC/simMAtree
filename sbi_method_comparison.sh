#!/bin/bash
# Script pour comparer différentes méthodes SBI sur un même jeu de données

# Vérifier les arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_config> <data_path> <output_dir>"
    echo "  model_config: Chemin vers le fichier de configuration du modèle"
    echo "  data_path: Chemin vers le fichier de données CSV existant"
    echo "  output_dir: Répertoire de sortie pour les résultats"
    exit 1
fi

MODEL_CONFIG=$1
DATA_PATH=$2
OUTPUT_DIR=$3

# Créer le répertoire de sortie
mkdir -p "$OUTPUT_DIR"

# Méthodes SBI à comparer
SBI_METHODS=("NPE" "NPSE" "NLE" "MNLE" "NRE_A" "NRE" "BNRE")


ROUNDS=("1")

# Charger les paramètres du modèle
MODEL_TYPE=$(cat "$MODEL_CONFIG" | grep -o '"class_name" : "[^"]*"' | cut -d '"' -f 4)

echo "Starting SBI method comparison..."
echo "Model: $MODEL_TYPE"
echo "Data: $DATA_PATH"

# Si le fichier de configuration du modèle contient des paramètres vrais, on les extrait
if grep -q "\"params\"" "$MODEL_CONFIG"; then
    echo "Extracting true parameters from model config..."
    PARAMS_FILE="$OUTPUT_DIR/true_params.json"
    echo "{" > "$PARAMS_FILE"
    grep -A 20 "\"params\"" "$MODEL_CONFIG" | sed -n '/params/,/}/p' | grep -v "params" | grep -v "^--$" >> "$PARAMS_FILE"
fi

# Pour chaque méthode SBI
for method in "${SBI_METHODS[@]}"; do
    # Pour chaque configuration de rounds
    for rounds in "${ROUNDS[@]}"; do
        echo
        echo "===== Testing $method with $rounds round(s) ====="
        
        # Créer le répertoire pour cette méthode et configuration
        METHOD_DIR="$OUTPUT_DIR/${method}_${rounds}rounds"
        mkdir -p "$METHOD_DIR"
        
        # Créer un fichier de configuration SBI temporaire
        SBI_CONFIG="$METHOD_DIR/sbi_config.json"
        cat > "$SBI_CONFIG" << EOF
{
    "module_name" : "inference.sbi_backend",
    "class_name" : "SbiBackend",
    "method" : "$method",
    "num_simulations" : 300,
    "num_rounds" : $rounds,
    "random_seed" : 42,
    "num_samples" : 200,
    "num_workers" : 10,
    "device" : "cpu"
}
EOF
        
        # Exécuter l'inférence avec cette méthode
        echo "Running inference with $method ($rounds rounds)..."
        python run.py --task inference \
                  --data_path "$DATA_PATH" \
                  --model_config "$MODEL_CONFIG" \
                  --inference_config "$SBI_CONFIG" \
                  --results_dir "$METHOD_DIR"
        
        # Évaluer les résultats si on a des paramètres vrais
        if [ -f "$PARAMS_FILE" ]; then
            echo "Evaluating results of $method ($rounds rounds)..."
            python run.py --task score \
                     --model_config "$MODEL_CONFIG" \
                     --results_dir "$METHOD_DIR" \
                     --true_params "$PARAMS_FILE"
        fi
    done
done

RESULTS_CSV="$OUTPUT_DIR/all_methods_results.csv"
echo "method,rounds,rmse,nrmse,mean_rel_error_pct,coverage_probability" > "$RESULTS_CSV"

# Collecter les métriques pour chaque méthode
for method in "${SBI_METHODS[@]}"; do
    for rounds in "${ROUNDS[@]}"; do
        METRICS_FILE="$OUTPUT_DIR/${method}_${rounds}rounds/summary_metrics.csv"
        if [ -f "$METRICS_FILE" ]; then
            # Extraire les valeurs des métriques
            METRICS=$(tail -n 1 "$METRICS_FILE")
            echo "$method,$rounds,$METRICS" >> "$RESULTS_CSV"
        fi
    done
done

echo "Comparison completed. Results saved to $OUTPUT_DIR"