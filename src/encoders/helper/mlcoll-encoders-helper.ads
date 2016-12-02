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

with Ada.Finalization;

with MLColl;
with MLColl.Encoders.BiRNN;

package MLColl.Encoders.Helper is
   
    type Encoding_Mode_Type is
      (Direct_Token_Embeddings,
       BiRNN_Embeddings, 
       Window_Based_Embeddings);

    type BiRNN_Network_Type is
      (BiRNN_RNN, 
       BiRNN_GRU, 
       BiRNN_GRU_NO_OUTPUT_LAYER,
       BiRNN_NONE);

    type Encoder_Type
      (Encoding_Mode      : Encoding_Mode_Type;
       BiRNN_Network      : BiRNN_Network_Type;
       Encoded_Token_Size : Positive) is new Ada.Finalization.Controlled with record

        case Encoding_Mode is
            when BiRNN_Embeddings =>
                case BiRNN_Network is
                    when BiRNN_RNN | BiRNN_GRU | BiRNN_GRU_NO_OUTPUT_LAYER =>
                        BiRNN_Model       : MLColl.Encoders.BiRNN.Model_Access_Type := null;
                        Input_Token_Size  : Positive;
                        
                    when BiRNN_NONE =>
                        null;
                end case;

            when Direct_Token_Embeddings =>
                null;
                
            when Window_Based_Embeddings =>
                null;
        end case;

    end record;

    overriding procedure Initialize (Object : in out Encoder_Type);
    overriding procedure Adjust     (Object : in out Encoder_Type);
    overriding procedure Finalize   (Object : in out Encoder_Type);

    Encoders_Helper_Error : exception;
    
    package BiRNN_Helper is
        
        procedure Initialize_BiRNN_Encoder
          (Encoder           : in out Encoder_Type;
           Input_Token_Size  : in     Positive;
           Hidden_Layer_Size : in     Positive);  
        
    end BiRNN_Helper;
    
end MLColl.Encoders.Helper;
