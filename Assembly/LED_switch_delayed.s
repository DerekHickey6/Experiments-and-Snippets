.equ SWITCH, 0xff200040
.equ LED, 0xff200000

.global _start
_start:
		
		LDR R1, =SWITCH
		LDR R2, =LED

		LDR R3, [R1]

LOOP:   
        MOV R5, #300   
		STR R3, [R2]
		MVN R3, R3 

OUTER:  
        MOV R4, #10000            

INNER:  
        SUBS R4, R4, #1     
        BNE INNER           

        SUBS R5, R5, #1     
        BNE OUTER           

        B LOOP              