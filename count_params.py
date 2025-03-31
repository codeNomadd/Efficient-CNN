from model import EfficientNetModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize the model
model = EfficientNetModel()
model_instance = model.get_model()

# Count parameters
total_params = count_parameters(model_instance)
print(f"Total number of parameters: {total_params:,}")
print(f"Total number of parameters (in millions): {total_params/1e6:.2f}M") 