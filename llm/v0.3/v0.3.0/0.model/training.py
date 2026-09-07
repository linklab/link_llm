"""v0.3.0 학습 루프: Adam, 토큰 단위 손실, 검증 최저 가중치 선택."""
import math
import torch


def fit(lm, sentences, valid_sentences=None):
    if lm.EPOCHS < 1 or lm.BATCH_SIZE < 1 or lm.LR <= 0:
        raise ValueError("epochs, batch_size, lr은 양수여야 합니다.")
    torch.manual_seed(lm.SEED)
    lm.initialize(sentences)
    rows = lm.make_windows(sentences)
    if not rows:
        raise ValueError("학습할 토큰이 없습니다.")
    for _, targets, known in rows:
        if any(t != -100 and not k for t, k in zip(targets, known)):
            raise ValueError("학습 데이터에 예약 PAD 또는 미등록 토큰이 있습니다.")
    loader = lm.batches(rows, lm.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(lm.SEED))
    valid_rows = lm.make_windows(valid_sentences) if valid_sentences else None
    if valid_sentences and not valid_rows:
        raise ValueError("검증할 토큰이 없습니다.")
    optimizer = torch.optim.Adam(lm.net.parameters(), lr=lm.LR)
    best, best_state, bad = float("inf"), None, 0
    lm.history = []
    for epoch in range(1, lm.EPOCHS + 1):
        lm.net.train()
        total, count = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(lm.device()), y.to(lm.device())
            optimizer.zero_grad(set_to_none=True)
            loss = lm.sequence_loss(lm.net(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("학습 손실이 유한하지 않습니다.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in lm.net.parameters()):
                raise FloatingPointError("gradient가 유한하지 않습니다.")
            optimizer.step()
            n = int(y.ne(-100).sum())
            total += loss.item() * n
            count += n
        train_loss = total / count
        valid = lm.evaluate_rows(valid_rows) if valid_rows is not None else None
        score = valid["ppl"] if valid else train_loss
        lm.history.append({"epoch": epoch, "train_loss": train_loss,
                           "valid_ppl": valid["ppl"] if valid else None})
        if score < best:
            best, bad = score, 0
            lm.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in lm.net.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d} loss={train_loss:.4f} "
                  f"valid_ppl={valid['ppl'] if valid else math.nan:.4f}", flush=True)
        if valid and lm.PATIENCE > 0 and bad >= lm.PATIENCE:
            break
    lm.net.load_state_dict(best_state)
    lm.net.eval()
    return lm.net
