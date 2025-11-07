"""
Visualização de Métricas de Treinamento ICNN

Gera relatório visual completo com:
- Curvas de loss (treino e teste)
- Análise de performance
- Distribuição de pesos (convexidade)
- Scatter plots de predições
- Relatório HTML
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include("../icnn/ICNN.jl")
using .ICNN

using Plots
using CSV
using DataFrames
using Statistics
using Printf
using JSON
using Dates

println("=" ^ 80)
println("VISUALIZAÇÃO DE MÉTRICAS DE TREINAMENTO")
println("=" ^ 80)
println()

# ============================================================================
# Configuração
# ============================================================================

experiment_dir = "/home/gabemed/purdue/ICNN-conterfatual/tmp/dcopf_experiment"
output_dir = joinpath(experiment_dir, "visualizations")
mkpath(output_dir)

println("Carregando métricas de: $experiment_dir")
println("Salvando visualizações em: $output_dir")
println()

# ============================================================================
# 1. Carregar Dados
# ============================================================================

println("1. Carregando dados...")
println("-" ^ 80)

# Training log (CSV)
log_file = joinpath(experiment_dir, "training_log.csv")
if !isfile(log_file)
    error("Training log não encontrado: $log_file")
end

df = CSV.read(log_file, DataFrame)
println("✓ Training log carregado: $(nrow(df)) epochs")

# Métricas (JSON)
metrics_file = joinpath(experiment_dir, "metrics_julia.json")
metrics = Dict()
if isfile(metrics_file)
    metrics = JSON.parsefile(metrics_file)
    println("✓ Métricas JSON carregadas")
else
    println("⚠ Métricas JSON não encontradas (opcional)")
end

# Modelo
model_file = joinpath(experiment_dir, "best_model.bson")
result = load_model(model_file)

model, scaler_X, scaler_Y = result
println("✓ Modelo carregado (com scalers)")
println()

# ============================================================================
# 2. Plot: Loss Curves
# ============================================================================

println("2. Gerando gráfico de Loss Curves...")
println("-" ^ 80)

p1 = plot(df.epoch, df.train_loss,
         label="Train Loss",
         linewidth=2,
         xlabel="Epoch",
         ylabel="MSE Loss",
         title="Training and Test Loss",
         legend=:topright,
         grid=true,
         yscale=:log10,
         size=(800, 500))

if "test_loss" in names(df)
    plot!(p1, df.epoch, df.test_loss,
          label="Test Loss",
          linewidth=2,
          linestyle=:dash)
end

savefig(p1, joinpath(output_dir, "1_loss_curves.png"))
println("✓ Salvo: 1_loss_curves.png")

# ============================================================================
# 3. Plot: Loss por Tempo
# ============================================================================

println("3. Gerando gráfico de Loss por Tempo...")
println("-" ^ 80)

# Calcular tempo cumulativo
cumulative_time = cumsum(df.time)

p2 = plot(cumulative_time, df.train_loss,
         label="Train Loss",
         linewidth=2,
         xlabel="Tempo (segundos)",
         ylabel="MSE Loss",
         title="Loss vs Training Time",
         legend=:topright,
         grid=true,
         yscale=:log10,
         size=(800, 500))

if "test_loss" in names(df)
    plot!(p2, cumulative_time, df.test_loss,
          label="Test Loss",
          linewidth=2,
          linestyle=:dash)
end

savefig(p2, joinpath(output_dir, "2_loss_vs_time.png"))
println("✓ Salvo: 2_loss_vs_time.png")

# ============================================================================
# 4. Análise de Performance
# ============================================================================

println("4. Gerando análise de performance...")
println("-" ^ 80)

# Carregar dataset para avaliar
data_path = "/home/gabemed/purdue/ICNN-conterfatual/icnn/data/data_pglib_opf_case118_ieee.bson"
dataset = prepare_dcopf_dataset(data_path; train_ratio=0.8, normalize_method=:none, seed=42)

# Predições
Y_pred_train = predict(model, dataset.X_train)
Y_pred_test = predict(model, dataset.X_test)

# Métricas
mse_train = mean((Y_pred_train .- dataset.Y_train).^2)
mse_test = mean((Y_pred_test .- dataset.Y_test).^2)
mae_train = mean(abs.(Y_pred_train .- dataset.Y_train))
mae_test = mean(abs.(Y_pred_test .- dataset.Y_test))
rmse_train = sqrt(mse_train)
rmse_test = sqrt(mse_test)

# R² score
function r2_score(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end

r2_train = r2_score(dataset.Y_train, Y_pred_train)
r2_test = r2_score(dataset.Y_test, Y_pred_test)

println("Performance Metrics:")
println("  Train: MSE=$(round(mse_train, digits=6)), RMSE=$(round(rmse_train, digits=6)), R²=$(round(r2_train, digits=4))")
println("  Test:  MSE=$(round(mse_test, digits=6)), RMSE=$(round(rmse_test, digits=6)), R²=$(round(r2_test, digits=4))")

# ============================================================================
# 5. Plot: Scatter Predictions
# ============================================================================

println("5. Gerando scatter plots de predições...")
println("-" ^ 80)

# Desnormalizar para escala original
Y_true_train_denorm = denormalize_output(dataset.Y_train, scaler_Y)
Y_pred_train_denorm = denormalize_output(Y_pred_train, scaler_Y)
Y_true_test_denorm = denormalize_output(dataset.Y_test, scaler_Y)
Y_pred_test_denorm = denormalize_output(Y_pred_test, scaler_Y)

# Scatter train
p3 = scatter(Y_true_train_denorm, Y_pred_train_denorm,
            alpha=0.3,
            markersize=3,
            label="Train",
            xlabel="True Cost (\$)",
            ylabel="Predicted Cost (\$)",
            title="Predictions vs True Values",
            legend=:topleft,
            size=(800, 600))

# Scatter test
scatter!(p3, Y_true_test_denorm, Y_pred_test_denorm,
         alpha=0.5,
         markersize=4,
         label="Test",
         color=:red)

# Linha ideal (y=x)
min_val = min(minimum(Y_true_train_denorm), minimum(Y_true_test_denorm))
max_val = max(maximum(Y_true_train_denorm), maximum(Y_true_test_denorm))
plot!(p3, [min_val, max_val], [min_val, max_val],
      linestyle=:dash,
      linewidth=2,
      color=:black,
      label="Perfect Prediction")

savefig(p3, joinpath(output_dir, "3_predictions_scatter.png"))
println("✓ Salvo: 3_predictions_scatter.png")

# ============================================================================
# 6. Plot: Histograma de Erros
# ============================================================================

println("6. Gerando histograma de erros...")
println("-" ^ 80)

errors_train = Y_pred_train_denorm .- Y_true_train_denorm
errors_test = Y_pred_test_denorm .- Y_true_test_denorm

p4 = histogram(errors_train[:],
              bins=50,
              alpha=0.5,
              label="Train",
              xlabel="Prediction Error (\$)",
              ylabel="Frequency",
              title="Distribution of Prediction Errors",
              legend=:topright,
              size=(800, 500))

histogram!(p4, errors_test[:],
          bins=50,
          alpha=0.5,
          label="Test")

vline!(p4, [0],
       linestyle=:dash,
       linewidth=2,
       color=:black,
       label="Zero Error")

savefig(p4, joinpath(output_dir, "4_error_distribution.png"))
println("✓ Salvo: 4_error_distribution.png")

# ============================================================================
# 7. Análise de Convexidade (Pesos)
# ============================================================================

println("7. Analisando convexidade da rede...")
println("-" ^ 80)

# Coletar todos os pesos das camadas escondidas
all_weights = Float64[]
layer_info = []

for (i, layer) in enumerate(model.hidden_layers)
    weights = vec(layer.weight)
    append!(all_weights, weights)
    
    n_negative = count(w -> w < 0, weights)
    min_w = minimum(weights)
    max_w = maximum(weights)
    mean_w = mean(weights)
    
    push!(layer_info, (
        layer=i,
        n_weights=length(weights),
        n_negative=n_negative,
        min=min_w,
        max=max_w,
        mean=mean_w
    ))
    
    println("  Layer $i: $(length(weights)) weights, $(n_negative) negative, range=[$(round(min_w, digits=6)), $(round(max_w, digits=6))]")
end

# Plot distribuição de pesos
p5 = histogram(all_weights,
              bins=100,
              xlabel="Weight Value",
              ylabel="Frequency",
              title="Distribution of Hidden Layer Weights (Convexity Check)",
              legend=false,
              size=(800, 500))

vline!(p5, [0],
       linestyle=:dash,
       linewidth=3,
       color=:red,
       label="Zero (Convexity Boundary)")

savefig(p5, joinpath(output_dir, "5_weight_distribution.png"))
println("✓ Salvo: 5_weight_distribution.png")

# ============================================================================
# 8. Relatório HTML
# ============================================================================

println("8. Gerando relatório HTML...")
println("-" ^ 80)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ICNN Training Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }
        .metric-box {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .metric-card.warning {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .success-box {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>🧠 ICNN Training Report</h1>
    
    <div class="info-box">
        <strong>Experiment:</strong> DCOPF Case 118 IEEE<br>
        <strong>Date:</strong> $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))<br>
        <strong>Normalization:</strong> $(scaler_Y[:method])<br>
        <strong>Dataset:</strong> $(size(dataset.X_train, 1)) train, $(size(dataset.X_test, 1)) test samples
    </div>

    <h2>📊 Performance Metrics</h2>
    
    <div class="metric-row">
        <div class="metric-card success">
            <div class="metric-label">Test R² Score</div>
            <div class="metric-value">$(round(r2_test * 100, digits=1))%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test RMSE</div>
            <div class="metric-value">$(round(rmse_test, digits=4))</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Test MAE</div>
            <div class="metric-value">$(round(mae_test, digits=4))</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Epochs Trained</div>
            <div class="metric-value">$(nrow(df))</div>
        </div>
    </div>

    <div class="metric-box">
        <h3>Detailed Performance</h3>
        <table>
            <tr>
                <th>Dataset</th>
                <th>MSE</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>R²</th>
            </tr>
            <tr>
                <td><strong>Train</strong></td>
                <td>$(round(mse_train, digits=6))</td>
                <td>$(round(rmse_train, digits=6))</td>
                <td>$(round(mae_train, digits=6))</td>
                <td>$(round(r2_train, digits=4))</td>
            </tr>
            <tr>
                <td><strong>Test</strong></td>
                <td>$(round(mse_test, digits=6))</td>
                <td>$(round(rmse_test, digits=6))</td>
                <td>$(round(mae_test, digits=6))</td>
                <td>$(round(r2_test, digits=4))</td>
            </tr>
        </table>
    </div>

    <h2>📈 Training Progress</h2>
    <img src="1_loss_curves.png" alt="Loss Curves">
    <img src="2_loss_vs_time.png" alt="Loss vs Time">

    <h2>🎯 Prediction Quality</h2>
    <img src="3_predictions_scatter.png" alt="Predictions Scatter">
    <img src="4_error_distribution.png" alt="Error Distribution">

    <h2>🔧 Convexity Analysis</h2>
    
    <div class="success-box">
        <strong>✓ Convexity Check:</strong> $(all(info.n_negative == 0 for info in layer_info) ? "PASSED - All weights non-negative" : "WARNING - Some negative weights detected")
    </div>

    <img src="5_weight_distribution.png" alt="Weight Distribution">

    <div class="metric-box">
        <h3>Layer-wise Weight Statistics</h3>
        <table>
            <tr>
                <th>Layer</th>
                <th># Weights</th>
                <th># Negative</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
            </tr>
"""

for info in layer_info
    global html_content
    html_content *= """
            <tr>
                <td>Hidden Layer $(info.layer)</td>
                <td>$(info.n_weights)</td>
                <td>$(info.n_negative)</td>
                <td>$(round(info.min, digits=6))</td>
                <td>$(round(info.max, digits=6))</td>
                <td>$(round(info.mean, digits=6))</td>
            </tr>
"""
end

html_content *= """
        </table>
    </div>

    <h2>📝 Model Configuration</h2>
    
    <div class="metric-box">
        <table>
            <tr>
                <td><strong>Input Features</strong></td>
                <td>$(model.n_features)</td>
            </tr>
            <tr>
                <td><strong>Output Dimension</strong></td>
                <td>$(model.n_output)</td>
            </tr>
            <tr>
                <td><strong>Hidden Layers</strong></td>
                <td>$(model.layers)</td>
            </tr>
            <tr>
                <td><strong>Total Parameters</strong></td>
                <td>$(sum(length(layer.weight) + length(layer.bias) for layer in model.hidden_layers) + length(model.input_layer.weight) + length(model.input_layer.bias))</td>
            </tr>
            <tr>
                <td><strong>Normalization Method</strong></td>
                <td>$(scaler_Y[:method])</td>
            </tr>
            <tr>
                <td><strong>Target Range (original)</strong></td>
                <td>$(if haskey(scaler_Y, :min)
                    "[\$$(round(scaler_Y[:min], digits=2)), \$$(round(scaler_Y[:max], digits=2))]"
                else
                    "[Not normalized - values in original scale]"
                end)</td>
            </tr>
        </table>
    </div>

    <h2>✅ Summary</h2>
    
    $(r2_test > 0.95 ? 
        "<div class=\"success-box\"><strong>EXCELLENT</strong> - Model achieves R² > 95% on test set. High-quality predictions.</div>" :
      r2_test > 0.85 ?
        "<div class=\"info-box\"><strong>GOOD</strong> - Model achieves R² > 85% on test set. Acceptable performance.</div>" :
        "<div class=\"warning-box\"><strong>NEEDS IMPROVEMENT</strong> - Model R² < 85%. Consider: more epochs, different architecture, or hyperparameter tuning.</div>")

    $(all(info.n_negative == 0 for info in layer_info) ?
        "<div class=\"success-box\"><strong>✓ CONVEXITY VERIFIED</strong> - All hidden layer weights are non-negative. ICNN convexity property satisfied.</div>" :
        "<div class=\"warning-box\"><strong>⚠ CONVEXITY ISSUE</strong> - Some negative weights detected. Convexity enforcement may need adjustment.</div>")

    <div class="footer">
        <p>Generated by ICNN Training Visualization Tool</p>
        <p>$(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))</p>
    </div>
</body>
</html>
"""

# Salvar HTML
html_file = joinpath(output_dir, "training_report.html")
open(html_file, "w") do f
    write(f, html_content)
end

println("✓ Salvo: training_report.html")
println()

# ============================================================================
# Resumo Final
# ============================================================================

println("=" ^ 80)
println("VISUALIZAÇÕES GERADAS")
println("=" ^ 80)
println()
println("Arquivos salvos em: $output_dir")
println()
println("  1. 1_loss_curves.png         - Curvas de loss (train/test)")
println("  2. 2_loss_vs_time.png        - Loss vs tempo de treinamento")
println("  3. 3_predictions_scatter.png - Scatter: predições vs true")
println("  4. 4_error_distribution.png  - Histograma de erros")
println("  5. 5_weight_distribution.png - Distribuição de pesos")
println("  6. training_report.html      - Relatório completo HTML")
println()
println("=" ^ 80)
println("ABRA O RELATÓRIO HTML NO NAVEGADOR:")
println("=" ^ 80)
println()
println("  file://$(html_file)")
println()
println("Ou execute:")
println("  xdg-open $(html_file)  # Linux")
println("  open $(html_file)      # Mac")
println()

