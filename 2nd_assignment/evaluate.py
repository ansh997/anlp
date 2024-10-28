import torch
from Old_dataset import get_dataloaders
from sacrebleu import corpus_bleu

from train import Transformer

def evaluate(model, dataloader, criterion, device, tgt_vocab):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            # Shift target tokens for teacher forcing
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            # Forward pass
            output = model(src_batch, tgt_input)

            # Flatten the output and target for cross-entropy loss
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

            # Convert model output to predicted token indices
            preds = output.argmax(dim=-1).view(src_batch.size(0), -1).tolist()
            refs = tgt_output.view(src_batch.size(0), -1).tolist()

            # Decode token indices to text
            preds_text = [tgt_vocab.decode(pred) for pred in preds]
            refs_text = [[tgt_vocab.decode(ref)] for ref in refs]  # BLEU expects a list of reference lists

            all_preds.extend(preds_text)
            all_refs.extend(refs_text)

    # Compute BLEU score
    bleu_score = corpus_bleu(all_preds, all_refs).score

    # Write all BLEU scores to a file
    with open("bleu_scores.txt", "w") as f:
        for pred, ref in zip(all_preds, all_refs):
            f.write(f"Prediction: {pred}\nReference: {ref}\n\n")

    return epoch_loss / len(dataloader), bleu_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    scratch_location = '/scratch/hmnshpl/anlp_data/ted-talks-corpus'
    src_lang = 'en'
    tgt_lang = 'fr'
    batch_size = 64
    max_len = 100
    _, _, test_loader = get_dataloaders(scratch_location, src_lang, tgt_lang, batch_size, max_len)

    # Load model
    model_dim = 512
    src_vocab_size = test_loader.dataset.src_vocab.__len__()
    tgt_vocab_size = test_loader.dataset.tgt_vocab.__len__()
    model = Transformer(src_vocab_size, tgt_vocab_size, model_dim).to(device)
    model.load_state_dict(torch.load('/home2/hmnshpl/projects/anlp/2nd_assignment/transformer.pt'))
    model.eval()

    # Criterion (loss function)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=test_loader.dataset.tgt_vocab.pad_idx).to(device)

    # Evaluate the model
    tgt_vocab = test_loader.dataset.tgt_vocab
    test_loss, bleu_score = evaluate(model, test_loader, criterion, device, tgt_vocab)

    print(f'Test Loss: {test_loss:.4f}, BLEU: {bleu_score:.4f}')
    print('BLEU scores saved to bleu_scores.txt')

if __name__ == "__main__":
    main()
