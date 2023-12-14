Train a GPT from scratch

This is a character level GPT.

Both torch's transformer encoder and decoer layers can be used as GPT layer. The encoder layer can directly be used since it has no cross attention layer and also accepts attention mask. Using the decoder layer as GPT layer requires modification to the cross attention layer.


Usage:  
Put source text in assets/source.txt

then:  
python gpt.py -dim_model 256 -num_layers 2 -decoder_backbone 1 


When trained on Plato's *The Republic*, here is a sample generated text of 1000 tokens using prompt "I need another story" :

*I need another story-tellers are guilty
of maintains in the likeness of so many strangers and in many and men,
and in no other poverty of the soul?*

*Yes, he replied.*   
*Would you agree with me in thinking that they do not only ask but
pleasure and pain and weakness, like yourself, will reply that to go our
artisans, or the person of shoes, or any other remaining unnatural power;
and when fault not the discussion which has been described.
Other figures, such as the way of 'the year which I am speaking, and which
cannot have been mentioned, and are always being done which of them is
the true philosopher.* 

*Exactly.*   
*Then the sun in knowledge which of the two classes should be the rulers,
and who are the worst and made them over which they show the power and shell
the change, and pity them?* 

*Certainly not.*   
*Then mad or intemperate pleasures and pains are generally found in children
and women are city of women, and will the process the soul, there is a
further stage of character which we w*
