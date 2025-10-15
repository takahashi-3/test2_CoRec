import config
import torch
import transformers
import torch.nn as nn
from torchcrf import CRF

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_ctag, num_tag):
        super(EntityModel, self).__init__()
        self.num_ctag = num_ctag
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH, return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        #self.out_ctag = nn.Linear(768, self.num_ctag)
        self.out_tag = nn.Linear(768+1, self.num_tag)
        self.crf = CRF(self.num_tag, batch_first=True)
    
    def forward(self, ids, mask, token_type_ids, target_ctag, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        s = o1.size()
        target_ctag_r = torch.reshape(target_ctag, (s[0], s[1], -1))
        #target_ttag_r = torch.reshape(target_ttag, (s[0], s[1], -1))

        o2 = torch.cat((o1, target_ctag_r), -1)

        bo_tag = self.bert_drop(o2)
        #tag = self.out_tag(bo_tag)

        #loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        #loss = loss_tag

        emissions = self.out_tag(bo_tag)
        log_likelihood, tag = self.crf(emissions, target_tag, reduction='mean'), self.crf.decode(emissions)

        loss = (-1) * log_likelihood

        return tag, loss
