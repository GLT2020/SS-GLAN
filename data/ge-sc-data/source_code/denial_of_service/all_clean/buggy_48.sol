pragma solidity ^0.5.11; 
library SafeMath {	
    function add(uint a, uint b) internal pure returns (uint c) {	
        c = a + b;
        require(c >= a);
    }
    function sub(uint a, uint b) internal pure returns (uint c) {	
        require(b <= a);
        c = a - b;
    }
    function mul(uint a, uint b) internal pure returns (uint c) {	
        c = a * b;
        require(a == 0 || c / a == b);
    }
    function div(uint a, uint b) internal pure returns (uint c) {	
        require(b > 0);
        c = a / b;
    }
}
contract ERC20Interface {
    function totalSupply() public view returns (uint);
bool public payedOut_unchk9 = false;
function withdrawLeftOver_unchk9() public {
        require(payedOut_unchk9);
        msg.sender.send(address(this).balance);
    }							
    function balanceOf(address tokenOwner) public view returns (uint balance);
function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
  }				
    function allowance(address tokenOwner, address spender) public view returns (uint remaining);
function bug_unchk19() public{
address payable addr_unchk19;
if (!addr_unchk19.send (10 ether) || 1==1)
	{revert();}
}	
    function transfer(address to, uint tokens) public returns (bool success);
function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  }
    function approve(address spender, uint tokens) public returns (bool success);
bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    }
    function transferFrom(address from, address to, uint tokens) public returns (bool success);
bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
  function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event Transfer(address indexed from, address indexed to, uint tokens);
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}
contract ApproveAndCallFallBack {
    function receiveApproval(address from, uint256 tokens, address token, bytes memory data) public;
function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }	
}
contract Owned {
  function withdrawBal_unchk41 () public{
	uint64 Balances_unchk41 = 0;
	msg.sender.send(Balances_unchk41);}
  address public owner;
  function bug_unchk42() public{
uint receivers_unchk42;
address payable addr_unchk42;
if (!addr_unchk42.send(42 ether))
	{receivers_unchk42 +=1;}
else
	{revert();}
}
  address public newOwner;
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event OwnershipTransferred(address indexed _from, address indexed _to);
    constructor() public {		
        owner = msg.sender;
    }
function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address _newOwner) public onlyOwner {
        newOwner = _newOwner;
    }
function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(owner, newOwner);	
        owner = newOwner;
        newOwner = address(0);
    }
function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
}
contract QurasToken is ERC20Interface, Owned {		
    using SafeMath for uint;
  function unhandledsend_unchk2(address payable callee) public {
    callee.send(5 ether);
  }
  string public symbol;
  function bug_unchk43() public{
address payable addr_unchk43;
if (!addr_unchk43.send (10 ether) || 1==1)
	{revert();}
}
  string public  name;
  function my_func_uncheck48(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  uint8 public decimals;
  function withdrawBal_unchk17 () public{
	uint64 Balances_unchk17 = 0;
	msg.sender.send(Balances_unchk17);}
  uint _totalSupply;			
  function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
  }
  mapping(address => uint) balances;
  function bug_unchk3(address payable addr) public
      {addr.send (42 ether); }
  mapping(address => mapping(address => uint)) allowed;
    constructor() public {		
        symbol = "XQC";
        name = "Quras Token";
        decimals = 8;
        _totalSupply = 88888888800000000;
        balances[owner] = _totalSupply;		
        emit Transfer(address(0), owner, _totalSupply);		
    }
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
    function totalSupply() public view returns (uint) {		
        return _totalSupply.sub(balances[address(0)]);
    }
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
    function balanceOf(address tokenOwner) public view returns (uint balance) {		
        return balances[tokenOwner];
    }
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function transfer(address to, uint tokens) public returns (bool success) {
        balances[msg.sender] = balances[msg.sender].sub(tokens);
        balances[to] = balances[to].add(tokens);
        emit Transfer(msg.sender, to, tokens);		
        return true;
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
        allowed[msg.sender][_spender] = allowed[msg.sender][_spender].add(_addedValue);
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
        uint oldValue = allowed[msg.sender][_spender];
        if (_subtractedValue > oldValue) {
            allowed[msg.sender][_spender] = 0;
        } else {
            allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
        }
        emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
    function approve(address spender, uint tokens) public returns (bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);		
        return true;
    }
function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
    function transferFrom(address from, address to, uint tokens) public returns (bool success) {
        balances[from] = balances[from].sub(tokens);
        allowed[from][msg.sender] = allowed[from][msg.sender].sub(tokens);
        balances[to] = balances[to].add(tokens);
        emit Transfer(from, to, tokens);		
        return true;
    }
bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
    function allowance(address tokenOwner, address spender) public view returns (uint remaining) {		
        return allowed[tokenOwner][spender];
    }
function UncheckedExternalCall_unchk40 () public
{  address payable addr_unchk40;
   if (! addr_unchk40.send (2 ether))  
      {
      }
	else
      {
      }
}
    function approveAndCall(address spender, uint tokens, bytes memory data) public returns (bool success) {
        allowed[msg.sender][spender] = tokens;
        emit Approval(msg.sender, spender, tokens);		
        ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, address(this), data);
        return true;
    }
bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
    function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns (bool success) {
        return ERC20Interface(tokenAddress).transfer(owner, tokens);
    }
function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
}