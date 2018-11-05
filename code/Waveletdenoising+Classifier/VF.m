function [Sensitivity ,Precision,ACC,MCC,tp,tn] = VF(true_labels,predict_labels)
 %%
 % BAC (Ballanced ACcuracy) = (Sensitivity + Specificity) / 2,
 % where Sensitivity = true_positive / (true_positive + false_negative)
 % and   Specificity = true_negative / (true_negative + false_positive) 
     tp = 0;
     fn = 0;
     tn = 0;
     fp = 0;
     for run = 1:numel(true_labels)
         if true_labels(run) == 1  
             if predict_labels(run) == 1
                 tp = tp + 1;
             else
                 fn = fn + 1;
             end
         else
             if predict_labels(run) == 2
                 tn = tn + 1;
             else
                 fp = fp + 1;
             end
         end
     end
     Sensitivity = tp / (tp + fn)
     %Specificity = tn / (tn + fp)
     Precision = tp / (tp+fp)
     ACC =(tp+tn)/(tp+fn+tn+fp)
     MCC=(tp*tn-fp*fn)/sqrt((tp+fp)*(tn+fn)*(tp+fn)*(tn+fp))
     tp
     tn
end