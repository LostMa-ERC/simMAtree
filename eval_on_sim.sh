#!/bin/bash
# Script pour évaluer les performances d'inférence sur différents jeux de paramètres

# Vérifier les arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_config> <inference_config> <output_dir> [n_replicates]"
    echo "  model_config: Chemin vers le fichier de configuration du modèle"
    echo "  inference_config: Chemin vers le fichier de configuration de l'inférence"
    echo "  output_dir: Répertoire de sortie pour les résultats"
    echo "  n_replicates: Nombre de réplications (défaut: 3)"
    exit 1
fi

MODEL_CONFIG=$1
INFERENCE_CONFIG=$2
OUTPUT_DIR=$3
N_REPLICATES=${4:-3}

# Créer le répertoire de sortie
mkdir -p "$OUTPUT_DIR"

# Charger les paramètres du modèle
MODEL_TYPE=$(cat "$MODEL_CONFIG" | grep -o '"class_name" : "[^"]*"' | cut -d '"' -f 4)


echo "Starting evaluation experiment with $N_REPLICATES replicates..."
echo "Model: $MODEL_TYPE"

# Exécuter les réplications
for ((i=1; i<=$N_REPLICATES; i++)); do
    echo "=== Running replicate $i/$N_REPLICATES ==="
    
    REPLICATE_DIR="$OUTPUT_DIR/replicate_$i"
    PARAMS_FILE="$OUTPUT_DIR/true_params_$i.json"

    mkdir -p "$REPLICATE_DIR"
    
    echo "\n1. Generating synthetic data..."
    DATA_PATH="$REPLICATE_DIR/synthetic_data.csv"
    python run.py --task generate \
                  --data_path "$DATA_PATH" \
                  --seed $i \
                  --model_config "$MODEL_CONFIG" | tee temp_output.txt

    # Extraire les paramètres JSON entre les balises
    sed -n '/PARAMETERS_JSON_START/,/PARAMETERS_JSON_END/p' temp_output.txt | grep -v "PARAMETERS_JSON_" > "$PARAMS_FILE"
    rm temp_output.txt
    
    # Exécuter l'inférence
    echo "\n2. Running inference..."
    python run.py --task inference \
                  --data_path "$DATA_PATH" \
                  --model_config "$MODEL_CONFIG" \
                  --inference_config "$INFERENCE_CONFIG" \
                  --results_dir "$REPLICATE_DIR"
    
    # Évaluer les résultats
    echo "\n3. Evaluating results..."
    python run.py --task score \
                 --model_config "$MODEL_CONFIG" \
                 --results_dir "$REPLICATE_DIR" \
                 --true_params "$PARAMS_FILE"
done

# Analyser les résultats
echo "=== Aggregating results across replicates ==="

# Collecter les métriques de toutes les réplications
echo "replicate,rmse,bias,std_dev,coverage_probability" > "$OUTPUT_DIR/all_metrics.csv"
for ((i=1; i<=$N_REPLICATES; i++)); do
    METRICS_FILE="$OUTPUT_DIR/replicate_$i/evaluation/summary_metrics.csv"
    if [ -f "$METRICS_FILE" ]; then
        # Extraire les valeurs et ajouter le numéro de réplication
        METRICS=$(tail -n 1 "$METRICS_FILE")
        echo "$i,$METRICS" >> "$OUTPUT_DIR/all_metrics.csv"
    fi
done

# Calculer les moyennes et écarts-types des métriques
echo "=== Summary of evaluation metrics across all replicates ==="
echo "Metric,Mean,Std,Min,Max"

# Utiliser awk pour calculer les statistiques
for METRIC in rmse bias std_dev coverage_probability; do
    STATS=$(awk -F, -v col="$METRIC" 'NR==1 {for (i=1; i<=NF; i++) if ($i == col) col_idx=i} 
            NR>1 {sum+=$col_idx; sumsq+=$col_idx*$col_idx; if (min=="") min=$col_idx; if (max=="") max=$col_idx; 
            if ($col_idx < min) min=$col_idx; if ($col_idx > max) max=$col_idx} 
            END {mean=sum/(NR-1); std=sqrt(sumsq/(NR-1) - mean*mean); printf "%s,%.6f,%.6f,%.6f,%.6f\n", col, mean, std, min, max}' "$OUTPUT_DIR/all_metrics.csv")
    echo "$STATS"
done | tee "$OUTPUT_DIR/summary_statistics.csv"

echo "Evaluation completed. Results saved to $OUTPUT_DIR"