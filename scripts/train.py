import logging

from gensim.models import FastText

def train_embeddings(corpus, model_path):
    """
    Trains and saves a FastText model from a corpus file
    """
    
    logging.info("Initializing embedding model...")
    nela_model = FastText(vector_size=300, sg=1)
    logging.info("Building vocabulary...")
    nela_model.build_vocab(corpus_file=corpus)
    logging.info("Begin training!")
    nela_model.train(
                     corpus_file=corpus, epochs=nela_model.epochs,
                     total_examples=nela_model.corpus_count, total_words=nela_model.corpus_total_words
                    )
    logging.info("Training successfully finished!")
    logging.info("Saving model...")
    nela_model.save(model_path)
    logging.info("Model saved!")
    
    
def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stop_flag=0
    steps = 0
    best_loss = 1000
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        #print(train_iter.shape)
        train_batch = train_iter.sample(frac=1)
        batch_counter = 0
        while batch_counter < train_iter.shape[0]:
            batch = train_batch.iloc[batch_counter:batch_counter+args.batch_size]
            batch_counter += args.batch_size
            feature, target = batch.title_token, batch.label
            
            target = torch.FloatTensor(labeling(target))
            target.shape
            #feature = feature.data.t()
            #target = target.data.sub(1)  # batch first, index align
            #if args.cuda:
            #    feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            #print(logit.shape)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_inteval == 0:
                #print(torch.max(logit, 1)[1])
                corrects = (torch.max(logit, 1)[1] == torch.max(target, 1)[1]).sum()
                accuracy = 100.0 * corrects/len(batch)
                print(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             len(batch)))
            if steps % args.test_inteval == 0:
                dev_loss = eval(dev_iter, model, args)
                #print(dev_loss)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                        #### NEW ####
                        save(model, "./cnn/snapshot/", "best", "model")
                        #### NEW ####
                else:
                    print(steps,last_step)
                    #print(best_loss)
                    if steps - last_step >= args.early_stop:
                        to_stop = steps-last_step
                        if to_stop >= args.early_stop:
                            print("Early stop!")
                            stop_flag=1
                            return
                        print(to_stop)
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, './cnn/snapshot', steps)
                
            if stop_flag:
                break
        if stop_flag:
            break
            
                   
def evaluate(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    
    data_batch = data_iter.sample(frac=1)
    batch_counter = 0
    
    while batch_counter < data_iter.shape[0]:
        batch = data_batch.iloc[batch_counter:batch_counter+args.batch_size]
        batch_counter += args.batch_size
        feature, target = batch.title_token, batch.label

        target = torch.FloatTensor(labeling(target))
        target.shape

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1] == torch.max(target, 1)[1]).sum()
        
            

    size = len(data_iter)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    
    #print(avg_loss)
    return avg_loss