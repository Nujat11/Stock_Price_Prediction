# Import the custom loss from your utils module
from src.utils.losses import DirectionalMSELoss

# Inside your training setup / initialization block:
model = BiLSTMModel(...) # Or TransformerModel(...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set custom directional loss function (alpha=0.5 can be tuned)
criterion = DirectionalMSELoss(alpha=0.5)

# Inside your Training Loop:
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     predictions = model(inputs)
#     loss = criterion(predictions, targets)  # <-- Custom Loss Applied
#     loss.backward()
#     optimizer.step()