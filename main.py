import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # Para graficar
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import seaborn as sns
from dataset import SignsDataset
from models import ANN

# Determinar el dispositivo (GPU, MPS, o CPU)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# Cargar el conjunto de datos
dataset = SignsDataset()

BATCH_SIZE = 8
SHUFFLE = False
EPOCHS = 150

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

# Verificar el tamaño de las imágenes y las etiquetas
for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Crear el modelo
model = ANN(input_nodes=30 * 42 * 3, features=15).to(device)

# Función de pérdida y optimizador
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Función de entrenamiento
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Función para evaluar el modelo
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    num_batches = len(dataloader)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted_labels = pred.argmax(1)
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)

            # Guardar las etiquetas y las predicciones para calcular métricas
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())

    avg_loss = test_loss / num_batches
    accuracy = correct / total

    # Calcular precisión, recall y matriz de confusión
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, cm

# Función para graficar la matriz de confusión
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Ciclo principal de entrenamiento
def run():
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    eval_precisions = []
    eval_recalls = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}\n-------------------------------")

        # Entrenar el modelo
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        train_losses.append(train_loss)

        # Evaluar el modelo después de cada época
        print(f"Evaluating after epoch {epoch + 1}...")
        eval_loss, eval_accuracy, eval_precision, eval_recall, cm = evaluate_model(train_dataloader, model, loss_fn)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)
        eval_precisions.append(eval_precision)
        eval_recalls.append(eval_recall)

        print(f"Average Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy * 100:.2f}%, Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}")
        print("-" * 50)

    # Guardar el modelo
    torch.save(model.state_dict(), "model2.pth")

    # Graficar los resultados
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    # Graficar pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, eval_losses, label="Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Graficar precisión
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in eval_accuracies], label="Evaluation Accuracy")
    plt.plot(epochs, [prec * 100 for prec in eval_precisions], label="Evaluation Precision")
    plt.plot(epochs, [rec * 100 for rec in eval_recalls], label="Evaluation Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Percentage (%)")
    plt.title("Accuracy, Precision, and Recall over Epochs")
    plt.legend()
    plt.tight_layout()

    plot_confusion_matrix(cm, dataset.labels)

    plt.show()


if __name__ == "__main__":
    run()
