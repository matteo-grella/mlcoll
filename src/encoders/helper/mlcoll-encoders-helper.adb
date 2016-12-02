------------------------------------------------------------------------------
--                               M L C O L L
--  M a c h i n e   L e a r n i n g   C o m p o n e n t   C o l l e c t i o n
--
--        Copyright 2009-2013 M. Grella, S. Cangialosi, E. Brambilla
--
--  This is free software; you can redistribute it and/or modify it under
--  terms of the GNU General Public License as published by the Free Software
--  Foundation; either version 2, or (at your option) any later version.
--  This software is distributed in the hope that it will be useful, but WITH
--  OUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
--  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
--  for more details. Free Software Foundation, 59 Temple Place - Suite
--  330, Boston, MA 02111-1307, USA.
--
--  As a special exception, if other files instantiate generics from this
--  unit, or you link this unit with other files to produce an executable,
--  this unit does not by itself cause the resulting executable to be
--  covered by the GNU General Public License. This exception does not
--  however invalidate any other reasons why the executable file might be
--  covered by the GNU Public License.
--
------------------------------------------------------------------------------

pragma License (Modified_GPL);

with MLColl.Encoders.BiRNN.RNN;
with MLColl.Encoders.BiRNN.GRU;
with MLColl.Encoders.BiRNN.GRU_No_Output_Layer;

package body MLColl.Encoders.Helper is
    
    overriding procedure Initialize 
      (Object : in out Encoder_Type) is
    begin
        case Object.BiRNN_Network is
            when BiRNN_RNN =>
                
                if Object.Encoding_Mode /= BiRNN_Embeddings then
                    raise Encoders_Helper_Error with
                      "attempt (1) to initialize Encoder_Type with BiRNN_Embeddings_Mode = """ &
                      Object.BiRNN_Network'Img & """ and Encoding_Mode = """ & Object.Encoding_Mode'Img & """";
                end if;
                
                Object.BiRNN_Model := new MLColl.Encoders.BiRNN.RNN.RNN_Model_Type;

            when BiRNN_GRU =>
                
                if Object.Encoding_Mode /= BiRNN_Embeddings then
                    raise Encoders_Helper_Error with
                      "attempt (2) to initialize Encoder_Type with BiRNN_Embeddings_Mode = """ &
                      Object.BiRNN_Network'Img & """ and Encoding_Mode = """ & Object.Encoding_Mode'Img & """";
                end if;
                
                Object.BiRNN_Model := new MLColl.Encoders.BiRNN.GRU.GRU_Model_Type;                
                
            when BiRNN_GRU_NO_OUTPUT_LAYER =>
                
                if Object.Encoding_Mode /= BiRNN_Embeddings then
                    raise Encoders_Helper_Error with
                      "attempt (2) to initialize Encoder_Type with BiRNN_Embeddings_Mode = """ &
                      Object.BiRNN_Network'Img & """ and Encoding_Mode = """ & Object.Encoding_Mode'Img & """";
                end if;
                
                Object.BiRNN_Model := new MLColl.Encoders.BiRNN.GRU_No_Output_Layer.GRU_Model_Type;                
                
            when BiRNN_NONE =>
                if Object.Encoding_Mode /= Direct_Token_Embeddings then
                    raise Encoders_Helper_Error with
                      "attempt (3) to initialize Parser_Model_Type with BiRNN_Embeddings_Mode = """ &
                      Object.BiRNN_Network'Img & """ and Encoding_Mode = """ & Object.Encoding_Mode'Img & """";
                end if;
        end case;
    end Initialize;

    overriding procedure Adjust (Object : in out Encoder_Type) is
    begin
        case Object.BiRNN_Network is
            when BiRNN_RNN =>
                Object.BiRNN_Model 
                  := new MLColl.Encoders.BiRNN.RNN.RNN_Model_Type'
                    (MLColl.Encoders.BiRNN.RNN.RNN_Model_Type (Object.BiRNN_Model.all));
                
            when BiRNN_GRU =>
                Object.BiRNN_Model 
                  := new MLColl.Encoders.BiRNN.GRU.GRU_Model_Type'
                    (MLColl.Encoders.BiRNN.GRU.GRU_Model_Type (Object.BiRNN_Model.all));

            when BiRNN_GRU_NO_OUTPUT_LAYER =>
                Object.BiRNN_Model 
                  := new MLColl.Encoders.BiRNN.GRU_No_Output_Layer.GRU_Model_Type'
                    (MLColl.Encoders.BiRNN.GRU_No_Output_Layer.GRU_Model_Type (Object.BiRNN_Model.all));
                
            when BiRNN_NONE =>
                null;
        end case;
    end Adjust;
    
    overriding procedure Finalize (Object : in out Encoder_Type) is
    begin
        case Object.BiRNN_Network is
            when BiRNN_RNN | BiRNN_GRU | BiRNN_GRU_NO_OUTPUT_LAYER =>
                MLColl.Encoders.BiRNN.Free (Object.BiRNN_Model);
                
            when BiRNN_NONE =>
                null;
        end case;
    end Finalize;
    
    package body BiRNN_Helper is
    
        procedure Initialize_BiRNN_Encoder
          (Encoder           : in out Encoder_Type;
           Input_Token_Size  : in     Positive;
           Hidden_Layer_Size : in     Positive) is
        begin
            case Encoder.Encoding_Mode is
                when Direct_Token_Embeddings =>
                    null;

                when Window_Based_Embeddings =>
                    raise Encoders_Helper_Error with "Window_Based_Embeddings not implemented";
                    
                when BiRNN_Embeddings =>
                    
                    Encoder.Input_Token_Size := Input_Token_Size;
                    
                    Encoder.BiRNN_Model.Initialize_BiRNN_Model
                      (Input_Layer_Size  => Encoder.Input_Token_Size,
                       Hidden_Layer_Size => Hidden_Layer_Size,
                       Output_Layer_Size => Encoder.Encoded_Token_Size,
                       Verbose           => True);

                    if Encoder.BiRNN_Network in BiRNN_RNN | BiRNN_GRU | BiRNN_GRU_NO_OUTPUT_LAYER then
                        Pragma Assert
                          (Check   => Encoder.BiRNN_Model.Output_Layer_Size = Encoder.Encoded_Token_Size,
                           Message => "Encoder.BiRNN_Model.Output_Layer_Size /= Encoder.Encoded_Token_Size");
                    end if;

            end case;
        end Initialize_BiRNN_Encoder;
    
    end  BiRNN_Helper;

    
end MLColl.Encoders.Helper;
