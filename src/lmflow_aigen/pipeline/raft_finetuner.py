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


        for step, prompts in enumerate(dataset.train_dataloader):
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
                score_index=executor.map(model.get_score,image_list,step_list,prompt_list)

            iterator=0
            for max_scores in score_index:
                training_prompts.append([max_scores[0],max_scores[1],image_list[iterator][max_scores[2]],prompt_list[iterator]])
                iterator+=1


        model.get_text_image_score()
