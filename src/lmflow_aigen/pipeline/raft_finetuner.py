import torch
import concurrent

class RaftFinetuner:

    def __init__(self, raft_args):
        self.raft_args = raft_args
        
    def raft_finetune(self, model, dataset):
        
        # Load Diffusion Model pipeline
        self.pipeline = model.load_model_pipeline(accelerator)
        
        #Load Score Model
        model.load_score_model(self.raft_args.clip_model_pretrianed_or_path)

        inference_dataloader=dataset.inference_dataloader(self.raft_args.inference_batch_size)
        training_prompts=[]
        for step, prompts in enumerate(inference_dataloader):
            prompts=prompts['text']
            # Generating Images
            images=self.pipeline( 
                prompts,
                num_images_per_prompt=self.raft_args.num_images_per_prompt,
                width=self.data_args.resolution,
                height = self.data_args.resolution,
                num_inference_steps=0,              
                guidance_scale=6).images
            image_list=[]
            
            # Appending images to the image_list
            for i in range(int(len(images)/self.raft_args.num_images_per_prompt)):
                image_list.append(images[i*self.raft_args.num_images_per_prompt:(i+1)*self.raft_args.num_images_per_prompt])
            torch.cuda.empty_cache()

            # Get Aesthetic scores and CLIP scores of images
            with concurrent.futures.ThreadPoolExecutor(max_workers= self.raft_args.max_workers) as executor:
                step_list=[i for i in range(step*self.raft_args.raft_batch_size,(step+1)*self.raft_args.raft_batch_size)]
                score_index=executor.map(model.get_score,image_list,step_list,prompts)

            for iterator,max_scores in enumerate(score_index):
                print(iterator)
                training_prompts.append([max_scores[0],image_list[iterator][max_scores[1]],prompts[iterator]])

        training_prompts=[row[1:3] for row in sorted(training_prompts,key=lambda x: (x[0]),reverse=True)[:1]]

        images = [row[0] for row in training_prompts]
        texts = [row[1] for row in training_prompts]

        dataset.prepare_finetune_dataset(images, texts)


