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

package body MLColl.Encoders.BiRNN is

    procedure Fill
      (V_In       : in     Real_Array_Access_Array;
       V1_Out     : in out Real_Array_Access_Array;
       V2_Rev_Out : in out Real_Array_Access_Array) is

        I_Inv : Index_Type;
    begin
        for I in V_In'Range loop
            I_Inv := V_In'Last - (I - V_In'First);

            for J in V_In (I)'Range loop
                V1_Out (I) (J)         := V_In (I) (J);
                V2_Rev_Out (I_Inv) (J) := V_In (I) (J);
            end loop;
        end loop;
    end Fill;

    procedure Partition
      (V_In   : in Real_Array_Access_Array;
       V1_Out : in Real_Array_Access_Array;
       V2_Out : in Real_Array_Access_Array) is

        Partition_Index : constant Index_Type
          := (Index_Type'First + Index_Type (V_In (V_In'First)'Length) - 1) / 2;

        I_Inv : Index_Type;
    begin

        for I in V_In'Range loop
            I_Inv :=  V_In'Last - (I - V_In'First);

            for J in V_In (I)'First .. V_In (I)'Last loop
                if J <= Partition_Index then
                    V1_Out (I) (J) := V_In (I) (J);
                else
                    V2_Out (I_Inv) ((J - Partition_Index) - 1) := V_In (I) (J);
                end if;
            end loop;
        end loop;

    end Partition;

    procedure Concatenate
      (V_Out  : in out Real_Array_Access_Array;
       V1_In  : in     Real_Array_Access_Array;
       V2_In  : in     Real_Array_Access_Array) is

        Partition_Index : constant Index_Type
          := (Index_Type'First + Index_Type (V_Out (V_Out'First)'Length) - 1) / 2;

        I_Inv : Index_Type;
    begin
        for I in V_Out'Range loop
            I_Inv  := V_Out'Last - (I - V_Out'First);

            for J in V_Out (I)'Range loop
                V_Out (I)(J) := (if J <= Partition_Index then V1_In (I) (J) else V2_In (I_Inv) ((J - Partition_Index) - 1));
            end loop;
        end loop;
    end Concatenate;

    procedure Merge
      (V_Out  : in out Real_Array_Access_Array;
       V1_In  : in     Real_Array_Access_Array;
       V2_In  : in     Real_Array_Access_Array) is
        I_Inv : Index_Type;
    begin
        for I in V_Out'Range loop
            I_Inv := V_Out'Last - (I - V_Out'First);

            for J in V_Out (I)'Range loop
                V_Out (I) (J) := V1_In (I) (J) + V2_In (I_Inv) (J);
            end loop;
        end loop;
    end Merge;

end MLColl.Encoders.BiRNN;
