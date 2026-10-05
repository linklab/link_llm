"""Assistant-only, already-shifted SFT targets; no optimizer or model training."""
IGNORE = -100
POLICIES = ('reject', 'drop-oldest-pairs')


def examples(template, record, context=256, overflow='reject'):
    if type(context) is not int or context < 1 or overflow not in POLICIES:
        raise ValueError('invalid context or overflow policy')
    messages = record['messages']
    template.encode_conversation(messages)  # Validate the entire source first.
    system = int(messages[0]['role'] == 'system')
    result = []
    for target, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        kept = list(range(target + 1))
        dropped = []
        while True:
            selected = [messages[i] for i in kept]
            encoded = template.encode_conversation(selected)
            # Only the source's final response predicts conversation EOS.
            final = target == len(messages) - 1
            tokens = encoded['input_ids'] if final else encoded['input_ids'][:-1]
            if len(tokens) - 1 <= context:
                break
            if overflow == 'reject' or len(kept) - system <= 2:
                raise ValueError(f"{record['id']} assistant[{target}] exceeds context={context}; response/user/system never sliced")
            dropped.extend(kept[system:system + 2])
            del kept[system:system + 2]
        span = encoded['message_spans'][-1]
        # logits at t predicts token t+1: never shift these labels again.
        labels = [tokens[i] if i >= span['content_start'] else IGNORE
                  for i in range(1, len(tokens))]
        result.append({'id': f"{record['id']}:assistant:{target}",
                       'conversation_id': record['id'], 'assistant_message_index': target,
                       'kept_message_indices': kept, 'dropped_message_indices': dropped,
                       'input_ids': tokens[:-1], 'labels': labels,
                       'supervised_tokens': sum(x != IGNORE for x in labels),
                       'predicts_eos': final})
    return result


def collate(rows, pad_id=0):
    if not rows:
        raise ValueError('cannot collate an empty batch')
    if pad_id != 0:
        raise ValueError('the shared template requires PAD=0')
    if any(not r['input_ids'] or len(r['input_ids']) != len(r['labels']) for r in rows):
        raise ValueError('nonempty aligned input/labels required')
    size = max(len(r['input_ids']) for r in rows)
    return {'input_ids': [r['input_ids'] + [pad_id]*(size-len(r['input_ids'])) for r in rows],
            'labels': [r['labels'] + [IGNORE]*(size-len(r['labels'])) for r in rows],
            'attention_mask': [[1]*len(r['input_ids']) + [0]*(size-len(r['input_ids'])) for r in rows]}


def response_loss(logits, labels):
    """Mean over supervised tokens, not examples. Returns loss and token count.

    A zero-target batch returns differentiable zero; caller MUST skip optimizer
    and scheduler steps (including weight decay) when count == 0.
    """
    import torch
    from torch.nn import functional as F
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError('expected logits[B,T,V] and long labels[B,T]')
    active = labels.ne(IGNORE)
    count = int(active.sum().item())
    if not count:
        return logits.reshape(-1)[:0].sum(), 0
    targets = labels[active]
    if bool(((targets < 0) | (targets >= logits.shape[-1])).any()):
        raise ValueError('target outside vocabulary')
    return F.cross_entropy(logits[active].float(), targets), count
