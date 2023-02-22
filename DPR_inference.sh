python dense_retriever.py \
	--model_file None \
	--qa_dataset nq_test   \
	--ctx_datatsets [dpr_wiki] \
	--encoded_ctx_files [None]\
	--out_file output/output.json
	

	# --model_file /home/local/anaconda3/envs/paper/DPR/donwloaded_data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp \
	# --qa_dataset nq_test   \
	# --ctx_datatsets [dpr_wiki] \
	# --encoded_ctx_files ['/home/local/anaconda3/envs/paper/DPR/donwloaded_data/downloads/data/retriever_results/nq/single-adv-hn/wikipedia_passages_*\' ]\
	# --out_file /home/local/anaconda3/envs/paper/DPR/output/output.json
	

