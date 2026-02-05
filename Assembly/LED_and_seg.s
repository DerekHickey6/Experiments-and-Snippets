.equ SWITCH, 0xff200040
.equ LED, 0xff200000
.equ SEVSEG, 0xff200020

.global _start
_start:
		
		LDR R0, =SEVSEG
		LDR R1, =SWITCH
		LDR R2, =LED

		LDR R3, [R1]

LOOP:   
        MOV R5, #300   
		STR R3, [R2]
		MVN R3, R3 
		MOV R6, #0x6f
		STR R6, [R0]

OUTER:  
        MOV R4, #10000            

INNER:  
        SUBS R4, R4, #1     
        BNE INNER           

        SUBS R5, R5, #1     
        BNE OUTER           

        B LOOP              