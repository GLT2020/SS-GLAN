pragma solidity ^0.5.1;
contract CareerOnToken {
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed a_owner, address indexed _spender, uint256 _value);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event OwnerChang(address indexed _old,address indexed _new,uint256 _coin_change);
  function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public totalSupply;  
  function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
  string public name;                   
  function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
  uint8 public decimals;               
  function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
  string public symbol;               
  function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
  address public owner;
  function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) public balances;
  function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) public allowed;
  function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
  bool isTransPaused=false;
    constructor(
        uint256 _initialAmount,
        uint8 _decimalUnits) public 
    {
        owner=msg.sender;
		if(_initialAmount<=0){
		    totalSupply = 100000000000000000;   
		    balances[owner]=totalSupply;
		}else{
		    totalSupply = _initialAmount;   
		    balances[owner]=_initialAmount;
		}
		if(_decimalUnits<=0){
		    decimals=2;
		}else{
		    decimals = _decimalUnits;
		}
        name = "CareerOn Chain Token"; 
        symbol = "COT";
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(
        address _to, 
        uint256 _value) public returns (bool success) 
    {
        assert(_to!=address(this) && 
                !isTransPaused &&
                balances[msg.sender] >= _value &&
                balances[_to] + _value > balances[_to]
        );
        balances[msg.sender] -= _value;
        balances[_to] += _value;
		if(msg.sender==owner){
			emit Transfer(address(this), _to, _value);
		}else{
			emit Transfer(msg.sender, _to, _value);
		}
        return true;
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(
        address _from, 
        address _to, 
        uint256 _value) public returns (bool success) 
    {
        assert(_to!=address(this) && 
                !isTransPaused &&
                balances[msg.sender] >= _value &&
                balances[_to] + _value > balances[_to] &&
                allowed[_from][msg.sender] >= _value
        );
        balances[_to] += _value;
        balances[_from] -= _value; 
        allowed[_from][msg.sender] -= _value;
        if(_from==owner){
			emit Transfer(address(this), _to, _value);
		}else{
			emit Transfer(_from, _to, _value);
		}
        return true;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool success) 
    { 
        assert(msg.sender!=_spender && _value>0);
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(
        address _owner, 
        address _spender) public view returns (uint256 remaining) 
    {
        return allowed[_owner][_spender];
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
	function changeOwner(address newOwner) public{
        assert(msg.sender==owner && msg.sender!=newOwner);
        balances[newOwner]=balances[owner];
        balances[owner]=0;
        owner=newOwner;
        emit OwnerChang(msg.sender,newOwner,balances[owner]);
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function setPauseStatus(bool isPaused)public{
        assert(msg.sender==owner);
        isTransPaused=isPaused;
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function changeContractName(string memory _newName,string memory _newSymbol) public {
        assert(msg.sender==owner);
        name=_newName;
        symbol=_newSymbol;
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function () external payable {
        revert();
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
}