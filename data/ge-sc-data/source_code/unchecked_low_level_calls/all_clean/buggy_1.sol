pragma solidity >=0.4.22 <0.6.0;
contract EIP20Interface {
    uint256 public totalSupply;
    function balanceOf(address _owner) public view returns (uint256 balance);
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool success);
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool success);
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address _owner, address _spender) public view returns (uint256 remaining);
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed _from, address indexed _to, uint256 _value);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}
contract HotDollarsToken is EIP20Interface {
    uint256 constant private MAX_UINT256 = 2**256 - 1;
  function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => uint256) public balances;
  function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
  mapping (address => mapping (address => uint256)) public allowed;
  function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
  string public name;                   
  function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
  uint8 public decimals;                
  function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
  string public symbol;                 
    constructor() public {
        totalSupply = 3 * 1e28;                        
        name = "HotDollars Token";                          
        decimals = 18;                           
        symbol = "HDS";
        balances[msg.sender] = totalSupply; 
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balances[msg.sender] >= _value);
        balances[msg.sender] -= _value;
        balances[_to] += _value;
        emit Transfer(msg.sender, _to, _value); 
        return true;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        uint256 allowance = allowed[_from][msg.sender];
        require(balances[_from] >= _value && allowance >= _value);
        balances[_to] += _value;
        balances[_from] -= _value;
        if (allowance < MAX_UINT256) {
            allowed[_from][msg.sender] -= _value;
        }
        emit Transfer(_from, _to, _value); 
        return true;
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address _owner) public view returns (uint256 balance) {
        return balances[_owner];
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value); 
        return true;
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }
function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
}