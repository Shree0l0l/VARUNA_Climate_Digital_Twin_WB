from app.utils.model_performance import evaluate_model


print("=" * 60)
print("VARUNA 1-DAY MODEL TEST")
print("=" * 60)

results = evaluate_model()

print("\nPrediction shape:")
print(results["y_pred"].shape)

print("\nActual shape:")
print(results["y_true"].shape)

print("\nVariables:")
print(results["variables"])

print("\nMask shape:")
print(results["mask"].shape)

print("\n========== MODEL METRICS ==========")

for key, value in results["metrics"].items():
    print(f"{key}: {value}")

print("\n========== TEST COMPLETE ==========")